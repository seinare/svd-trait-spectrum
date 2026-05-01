import torch
import json
import argparse
import os
import re
import shutil
import tempfile
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from vllm import LLM, SamplingParams

# ======== SVD PERTURBATION ========
def apply_svd_energy_matthew(model, alpha=0.0):
    if alpha == 0.0: return
    print(f"Applying SVD: alpha={alpha} on ['up_proj', 'down_proj']")
    with torch.no_grad():
        for layer in tqdm(model.model.layers, desc="SVD"):
            for name in ["up_proj", "down_proj"]:
                proj = getattr(layer.mlp, name)
                W = proj.weight.data.float()
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)
                L = torch.log(torch.clamp(S, min=1e-9))
                M = torch.mean(L)
                L_new = L + alpha * (L - M)
                S_new = torch.exp(L_new)
                W_new = (U @ torch.diag(S_new) @ Vh).to(torch.bfloat16)
                proj.weight.data.copy_(W_new)

# ======== AST EVALUATOR ========
def extract_json_blocks(text):
    blocks = []
    # Try to extract from markdown blocks
    for match in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL):
        blocks.append(match.group(1))
    
    if not blocks:
        # Fallback to finding outermost brackets if no markdown
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            blocks.append(text[start:end+1])
    return blocks

def check_arguments_match(pred_args, gt_args):
    # gt_args is a dict of lists of possible values
    for k, possible_vals in gt_args.items():
        if k not in pred_args:
            # If the parameter is missing but "" is a valid answer, it's fine
            if "" in possible_vals:
                continue
            return False
        
        pred_val = pred_args[k]
        # Match type and value
        match = False
        for v in possible_vals:
            if str(pred_val).lower().strip() == str(v).lower().strip() or (v == "" and pred_val is None):
                match = True
                break
        if not match:
            return False
            
    # Check for hallucinated extra arguments
    for k in pred_args:
        if k not in gt_args:
            return False
            
    return True

def evaluate_ast(output_text, ground_truth):
    blocks = extract_json_blocks(output_text)
    
    res = {
        "arguments_valid_json": 0,
        "tool_name_correct": 0,
        "arguments_schema_valid": 0,
        "arguments_semantically_correct": 0,
        "extra_tool_calls": 0,
        "strict_success": 0
    }
    
    if not blocks:
        return res
        
    res["arguments_valid_json"] = 1
    
    try:
        parsed_blocks = [json.loads(b) for b in blocks]
    except:
        res["arguments_valid_json"] = 0
        return res
        
    if len(parsed_blocks) > 1:
        res["extra_tool_calls"] = 1
        
    call = parsed_blocks[0]
    
    # Ground truth is a list of possible correct tool call mappings
    # e.g. [{"calculate_triangle_area": {"base": [10], "height": [5]}}]
    
    # In simple setting, we just check if it matches ANY of the ground truths
    for gt_item in ground_truth:
        for expected_name, expected_args in gt_item.items():
            
            # Check name
            pred_name = call.get("name", "")
            if pred_name == expected_name:
                res["tool_name_correct"] = 1
            else:
                continue
                
            pred_args = call.get("arguments", {})
            if isinstance(pred_args, str):
                try:
                    pred_args = json.loads(pred_args)
                except:
                    pass
                    
            if not isinstance(pred_args, dict):
                continue
                
            res["arguments_schema_valid"] = 1
            
            if check_arguments_match(pred_args, expected_args):
                res["arguments_semantically_correct"] = 1
                
            # If we found a full match, we can stop
            if res["arguments_semantically_correct"] == 1:
                break
        if res["arguments_semantically_correct"] == 1:
            break
            
    if (res["tool_name_correct"] == 1 and 
        res["arguments_valid_json"] == 1 and 
        res["arguments_schema_valid"] == 1 and 
        res["arguments_semantically_correct"] == 1 and 
        res["extra_tool_calls"] == 0):
        res["strict_success"] = 1
        
    return res

def format_tool_prompt(query, tools):
    sys = "You are a helpful assistant with access to the following functions. Use them if required.\\n" + json.dumps(tools, indent=2) + "\\n\\nWhen calling a function, output a JSON block like:\\n```json\\n{\\n  \"name\": \"function_name\",\\n  \"arguments\": {\\n    \"arg1\": \"value1\"\\n  }\\n}\\n```"
    return sys + "\\n\\nUser: " + query + "\\nAssistant:"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.6)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--output_root", type=str, default="results/bfcl/llama1b_bfcl_ast")
    parser.add_argument("--tmp_root", type=str, default="/tmp")
    parser.add_argument("--prepared_model_dir", type=str, default=None)
    parser.add_argument("--save_prepared_model_dir", type=str, default=None)
    parser.add_argument("--local_files_only", action="store_true")
    args = parser.parse_args()

    model_id = args.model_id
    model_name_safe = model_id.split("/")[-1].replace(".", "_")
    device = f"cuda:{args.gpu}"
    out_path = os.path.join(args.output_root, f"{model_name_safe}_res_alpha{args.alpha}.json")
    if not args.smoke and os.path.exists(out_path):
        print("Exists")
        return

    tokenizer_source = args.prepared_model_dir or model_id
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, local_files_only=args.local_files_only)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    cleanup_dir = None
    if args.prepared_model_dir:
        eval_model_dir = args.prepared_model_dir
    elif args.alpha == 0.0 and not args.save_prepared_model_dir:
        eval_model_dir = model_id
    else:
        eval_model_dir = args.save_prepared_model_dir or tempfile.mkdtemp(
            prefix=f"bfcl_ast_{model_name_safe}_alpha{args.alpha}_",
            dir=args.tmp_root,
        )
        if not args.save_prepared_model_dir:
            cleanup_dir = eval_model_dir
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
            local_files_only=args.local_files_only,
        )
        apply_svd_energy_matthew(model, args.alpha)
        os.makedirs(eval_model_dir, exist_ok=True)
        model.save_pretrained(eval_model_dir)
        tokenizer.save_pretrained(eval_model_dir)
        del model
        torch.cuda.empty_cache()
    
    print("Loading BFCL V3 dataset...")
    from huggingface_hub import hf_hub_download
    questions_file = hf_hub_download(repo_id="gorilla-llm/Berkeley-Function-Calling-Leaderboard", filename="BFCL_v3_simple.json", repo_type="dataset")
    answers_file = hf_hub_download(repo_id="gorilla-llm/Berkeley-Function-Calling-Leaderboard", filename="possible_answer/BFCL_v3_simple.json", repo_type="dataset")
    
    with open(questions_file, "r") as f:
        questions = [json.loads(line) for line in f if line.strip()]
    with open(answers_file, "r") as f:
        answers = [json.loads(line) for line in f if line.strip()]
        
    # Map ID to ground truth
    gt_map = {row["id"]: row["ground_truth"] for row in answers}
    
    default_samples = 2 if args.smoke else min(200, len(questions))
    num_samples = args.num_samples if args.num_samples is not None else default_samples
    num_samples = min(num_samples, len(questions))
    items = []
    for row in questions[:num_samples]:
        q = row["question"]
        if isinstance(q, list) and len(q) > 0 and isinstance(q[0], list) and len(q[0]) > 0 and "content" in q[0][0]:
            q = q[0][0]["content"]
        elif isinstance(q, list) and len(q) > 0 and "content" in q[0]:
            q = q[0]["content"]
        else:
            q = str(q)
            
        items.append({
            "id": row["id"],
            "query": str(q), 
            "tools": row["function"],
            "ground_truth": gt_map[row["id"]]
        })

    print("Generating responses with retry limit=3...")
    llm = LLM(
        model=eval_model_dir,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
    )

    pass_at_1 = 0
    recovery_success = 0

    try:
        for item in tqdm(items, desc="Processing LLM calls"):
            prompt = format_tool_prompt(item["query"], item["tools"])
            history = prompt

            final_out = ""
            success_achieved = False

            for attempt in range(3):
                sampling_params = SamplingParams(temperature=0.0, max_tokens=256)
                outputs = llm.generate([history], sampling_params, use_tqdm=False)
                out_text = outputs[0].outputs[0].text.strip()
                final_out = out_text

                eval_res = evaluate_ast(out_text, item["ground_truth"])

                if eval_res["strict_success"] == 1:
                    success_achieved = True
                    if attempt == 0:
                        pass_at_1 += 1
                    recovery_success += 1
                    item["eval"] = eval_res
                    break

                err_msg = ""
                if eval_res["arguments_valid_json"] == 0:
                    err_msg = "Your output is not a valid JSON function call. Please retry and output strictly valid JSON."
                elif eval_res["tool_name_correct"] == 0:
                    err_msg = "You called an incorrect tool name. Please check the available tools and try again."
                else:
                    err_msg = "Your arguments do not match the required schema or semantic meaning. Please fix."

                history += "\\n" + out_text + "\\n\\nUser: " + err_msg + "\\nAssistant:"

            if not success_achieved:
                item["eval"] = evaluate_ast(final_out, item["ground_truth"])
    finally:
        from vllm.distributed.parallel_state import destroy_model_parallel
        destroy_model_parallel()
        del llm
        torch.cuda.empty_cache()
        if cleanup_dir:
            shutil.rmtree(cleanup_dir, ignore_errors=True)
        
    final_res = {
        "config": {
            "method": "energy_matthew_ast",
            "alpha": args.alpha,
            "model_id": model_id,
            "num_samples": len(items),
            "tensor_parallel_size": args.tensor_parallel_size,
            "max_model_len": args.max_model_len,
            "enforce_eager": args.enforce_eager,
            "prepared_model_dir": args.prepared_model_dir,
        },
        "metrics": {
            "First_pass_success": pass_at_1 / len(items),
            "Retry_enabled_success": recovery_success / len(items)
        }
    }
    
    print(f"Results for alpha={args.alpha}:")
    print(json.dumps(final_res, indent=2))
    
    os.makedirs(args.output_root, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(final_res, f, indent=2)

if __name__ == "__main__":
    main()

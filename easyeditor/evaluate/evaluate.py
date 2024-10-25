"""
Contains evaluation utilities for pytorch-based rewriting methods.
To use, simply call `compute_rewrite_quality_zsre` with the
appropriate arguments, which returns a dictionary containing them.
"""

import typing
from itertools import chain
from typing import List, Optional

import numpy as np
import torch
# from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer
from ..util import HyperParams
from .portability_evaluate import compute_portability_quality
from .evaluate_utils import (
    test_seq2seq_batch_prediction_acc, 
    test_batch_prediction_acc, 
    test_prediction_acc,
    test_generation_quality, 
    PPL,
    kl_loc_loss,
    es_sent
)

def compute_edit_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    record: typing.Dict,
    device,
    eval_metric: str = 'token_em',
    test_generation = False
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :paran snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """

    # First, unpack rewrite evaluation record.
    target_new, ground_truth = (
        record[x] for x in ["target_new", "ground_truth"]
    )

    rewrite_prompts = record["prompt"]
    rephrase_prompts = record["rephrase_prompt"] if 'rephrase_prompt' in record.keys() else None
    ret = compute_rewrite_or_rephrase_quality(model, model_name, hparams, tok,
                                              rewrite_prompts, target_new, device=device, eval_metric=eval_metric)

    ret['locality'] = {}
    ret['portability'] = {}
    if rephrase_prompts is not None:
        ret.update(
            compute_rewrite_or_rephrase_quality(model, model_name, hparams, tok,
                                                rephrase_prompts, target_new, device=device, test_rephrase=True, eval_metric=eval_metric)
        )

    if 'locality' in record.keys() and any(record['locality']):
        for locality_key in record['locality'].keys():
            ret['locality'].update(
                compute_locality_quality(model, model_name, hparams, tok, locality_key,
                                         record['locality'][locality_key]['prompt'],
                                         record['locality'][locality_key]['ground_truth'], device=device)
            )
    if 'portability' in record.keys() and any(record['portability']):
        for portability_key in record['portability'].keys():
            ret['portability'].update(
                compute_portability_quality(model, model_name, hparams, tok, portability_key,
                                            record['portability'][portability_key]['prompt'],
                                            record['portability'][portability_key]['ground_truth'], device=device)
            )
    if  test_generation:
        ret['fluency'] = test_generation_quality(model=model,tok=tok,prefixes=rewrite_prompts if isinstance(rewrite_prompts,list) else [rewrite_prompts,], max_out_len=100)
    return ret

def compute_rewrite_or_rephrase_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    prompt: str,
    target_new: str,
    device,
    test_rephrase: bool = False,
    eval_metric: str = 'token_em'
) -> typing.Dict:
    
    if not test_rephrase:
        key = 'rewrite'
    else:
        key = 'rephrase'
    if eval_metric == 'ppl':
        ppl = PPL(model, tok, prompt, target_new, device)
        ret = {
            f"{key}_ppl": ppl
        }
    else:
        if 't5' in model_name.lower():
            acc = test_seq2seq_batch_prediction_acc(model, tok, hparams, prompt, target_new, device)
        else:
            acc = test_prediction_acc(model, tok, hparams, prompt, target_new, device)
        ret = {
            f"{key}_acc": acc
        }
    return ret

def compute_locality_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    locality_key: str,
    prompt: str,
    locality_ground_truth: str,
    device,
) -> typing.Dict:

    if 't5' in model_name.lower():
        loc_tokens = test_seq2seq_batch_prediction_acc(model, tok, hparams, prompt, locality_ground_truth, device, locality=True)
    else:
        loc_tokens = test_prediction_acc(model, tok, hparams, prompt, locality_ground_truth, device, locality=True)

    if type(loc_tokens) is not list:
        loc_tokens = [loc_tokens,]

    ret = {
        f"{locality_key}_output": loc_tokens
    }
    return ret

def compute_icl_edit_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    icl_examples,
    record: typing.Dict,
    device,
    pre_edit: bool = False
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :param snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """

    # First, unpack rewrite evaluation record.
    target_new, ground_truth = (
        record[x] for x in ["target_new", "ground_truth"]
    )
    prompt = record["prompt"]

    rephrase = record["rephrase_prompt"] if 'rephrase_prompt' in record.keys() else None
    new_fact = f'New Fact: {prompt} {target_new}\nPrompt: {prompt}'

    if pre_edit:
        edit_acc = icl_lm_eval(model, model_name, hparams, tok, icl_examples,
                                       target_new, prompt)
    else:
        edit_acc = icl_lm_eval(model, model_name, hparams, tok, icl_examples,
                                              target_new, new_fact)
    ret = {
        f"rewrite_acc": edit_acc
    }
    ret['locality'] = {}
    ret['portability'] = {}
    if rephrase is not None:
        rephrase_acc = icl_lm_eval(model, model_name, hparams, tok, icl_examples,
                               target_new, f'New Fact: {prompt} {target_new}\nPrompt: {rephrase}')
        ret['rephrase_acc'] = rephrase_acc


    if 'locality' in record.keys() and any(record['locality']):
        for locality_key in record['locality'].keys():
            pre_neighbor = icl_lm_eval(model, model_name, hparams, tok, [''], record['locality'][locality_key]['ground_truth'],
                                       f"New Fact: {prompt} {target_new}\nPrompt: {record['locality'][locality_key]['prompt']}", neighborhood=True)
            post_neighbor = icl_lm_eval(model, model_name, hparams, tok, icl_examples, record['locality'][locality_key]['ground_truth'],
                                        f"New Fact: {prompt} {target_new}\nPrompt: {record['locality'][locality_key]['prompt']}", neighborhood=True)
            if type(pre_neighbor) is not list:
                pre_neighbor = [pre_neighbor, ]
            if type(post_neighbor) is not list:
                post_neighbor = [post_neighbor, ]
            assert len(pre_neighbor) == len(post_neighbor)

            ret['locality'][f'{locality_key}_acc'] = np.mean(np.equal(pre_neighbor, post_neighbor))
    # Form a list of lists of prefixes to test.
    if 'portability' in record.keys() and any(record['portability']):
        for portability_key in record['portability'].keys():
            if pre_edit:
                portability_acc = icl_lm_eval(model, model_name, hparams, tok, icl_examples, record['portability'][portability_key]['ground_truth'],
                                              record['portability'][portability_key]['prompt'])
            else:
                portability_acc = icl_lm_eval(model, model_name, hparams, tok, icl_examples, record['portability'][portability_key]['ground_truth'],
                                              f"New Fact: {prompt} {target_new}\nPrompt: {record['portability'][portability_key]['prompt']}")
            ret['portability'][f'{portability_key}_acc'] = portability_acc
    return ret

def icl_lm_eval(
        model,
        model_name,
        hparams: HyperParams,
        tokenizer,
        icl_examples,
        target,
        x,
        neighborhood=False
)-> typing.Dict:
    device = torch.device(f'cuda:{hparams.device}')
    if 't5' in model_name.lower():
        target_len = len(tokenizer.encode(target))
        target_ids = tokenizer(f'{x} {target}', return_tensors='pt')['input_ids'].to(device)
        encodings = tokenizer(''.join(icl_examples), return_tensors='pt')
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=target_ids).logits
            ans = torch.argmax(logits, dim=-1)[:,-target_len:-1].squeeze()
            target_ids = target_ids[:,-target_len:-1]
            if neighborhood:
                return ans.squeeze().detach().cpu().numpy().tolist()
            return torch.mean((ans == target_ids.to(ans.device).squeeze()).float(), dim=-1).detach().cpu().numpy().tolist()
    elif 'llama' in model_name.lower():
        target_ids = tokenizer(target, return_tensors='pt')['input_ids'].to(device)
        encodings = tokenizer(''.join(icl_examples) + f'{x} {target}', return_tensors='pt')
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        ans = torch.argmax(logits, dim=-1)[:,-target_ids.size(1):-1].squeeze()
        target_ids = target_ids[:,1:]   
        if neighborhood:
            return ans.squeeze().detach().cpu().numpy().tolist()
        return torch.mean((ans == target_ids.to(ans.device).squeeze()).float(), dim=-1).detach().cpu().numpy().tolist()        
    else:
        target_ids = tokenizer(' ' + target + '\n', return_tensors='pt')['input_ids'].to(device)
        encodings = tokenizer(''.join(icl_examples) + f'{x} {target}', return_tensors='pt')
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        ans = torch.argmax(logits, dim=-1)[:,-target_ids.size(1):-1].squeeze()
        target_ids = target_ids[:,:-1]
        if neighborhood:
            return ans.squeeze().detach().cpu().numpy().tolist()
        return torch.mean((ans == target_ids.to(ans.device).squeeze()).float(), dim=-1).detach().cpu().numpy().tolist()


def process_predict(logits, labels, tok):

    mask = labels != -100
    labels[~mask] = 0
    # 获取预测ID并根据掩码填充
    pred_ids = logits.argmax(-1).masked_fill(~mask, 0).detach().cpu()
    
    # 解码预测ID为字符串
    predict = tok.decode(pred_ids.tolist()[0], skip_special_tokens=True)
    return predict




def compute_icl_multimodal_edit_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    # vis_tok,
    icl_examples,
    record: typing.Dict,
    device,
    pre_edit: bool = False
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :param snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """
    vis_root = hparams.coco_image
    rephrase_root = hparams.rephrase_image



    # First, unpack rewrite evaluation record.
    target = record["target"]
    prompt = record["prompt"]
    image = record['image'].to(hparams.device) if torch.is_tensor(record['image']) and not record['image'].is_cuda else record['image']
    rephrase = record["rephrase_prompt"] if 'rephrase_prompt' in record.keys() else None
    rephrase_image = record["image_rephrase"] if 'image_rephrase' in record.keys() else None
    if rephrase_image is not None:
        rephrase_image = rephrase_image.to(hparams.device) if torch.is_tensor(rephrase_image) and not rephrase_image.is_cuda else rephrase_image
    ret = {}


    ###############################################################################
    knowledge_type = record["knowledge_type"]

    if knowledge_type ==0 or knowledge_type ==1:
        rel_prompt_1 = record["rel_prompt_1"]
        rel_ground_truth_1 = record["rel_ground_truth_1"]
        rel_prompt_2 = record["rel_prompt_2"]
        rel_ground_truth_2 = record["rel_ground_truth_2"]
        
        m_rel_prompt_1 = record["m_rel_prompt_1"]
        m_rel_ground_truth_1 = record["m_rel_ground_truth_1"]
        m_rel_prompt_2 = record["m_rel_prompt_2"]
        m_rel_ground_truth_2 = record["m_rel_ground_truth_2"]
    elif knowledge_type ==2:
        rel_prompt = record["rel_prompt"]
        rel_ground_truth = record["rel_ground_truth"]
       
        m_rel_prompt = record["m_rel_prompt"]
        m_rel_ground_truth = record["m_rel_ground_truth"]

        image_rephrase_question = record["image_rephrase_question"]
        one_hop_img = record['one_hop_img'].to(hparams.device) if torch.is_tensor(record['one_hop_img']) and not record['one_hop_img'].is_cuda else record['one_hop_img']
    ###############################################################################

    ###############################################################
    if "locality_prompt" in record.keys():
        loc_q = record["locality_prompt"]
        loc_a = record["locality_ground_truth"]
    if "multimodal_locality_image" in record.keys():
        m_loc_image = record['multimodal_locality_image'].to(hparams.device) if torch.is_tensor(record['multimodal_locality_image']) and not record['multimodal_locality_image'].is_cuda else record['multimodal_locality_image']
        m_loc_q = record["multimodal_locality_prompt"]
        m_loc_a = record["multimodal_locality_ground_truth"]
    
    new_fact = f'New Fact: {prompt} {target}\nPrompt: {prompt}'

    if pre_edit:
        edit_acc, _ = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,
                                       target, prompt, image)
    else:
        edit_acc, _ = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,
                                              target, new_fact, image)
        
    ret = {
        f"rewrite_acc": edit_acc
    }

    ret['rewrite_acc_prompt'] = record["prompt"]
    ret['rewrite_acc_ground_truth']  = record["target"]
    ret['rewrite_acc_predict'] = icl_multimodal_decode(model, model_name, hparams, tok, icl_examples, target, new_fact, None)
    
    
    if rephrase is not None:
        rephrase_acc, _ = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,
                               target, f'New Fact: {prompt} {target}\nPrompt: {rephrase}', image)
        ret['rephrase_acc'] = rephrase_acc

    ret['rephrase_acc_prompt'] = rephrase
    ret['rephrase_acc_ground_truth']  = record["target"]
    ret['rephrase_acc_predict'] = icl_multimodal_decode(model, model_name, hparams, tok, icl_examples, target, f'New Fact: {prompt} {target}\nPrompt: {rephrase}', None)

        
    if "image_rephrase" in record.keys():
        rephrase_image_acc, _ = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,
                               target, new_fact, rephrase_image)
        ret['rephrase_image_acc'] = rephrase_image_acc
    
    ret['rephrase_image_acc_prompt']   = record["prompt"]
    ret['rephrase_image_acc_ground_truth']  = record["target"]
    ret['rephrase_image_acc_predict']  = icl_multimodal_decode(model, model_name, hparams, tok, icl_examples, target, new_fact, None)
    
    ret["memory_alloc_max"] = torch.cuda.max_memory_allocated()
    ret["memory_res_max"] = torch.cuda.max_memory_reserved()


    if not pre_edit:
        if "locality_prompt" in record.keys():
            pre_text_loc_logits = icl_multimodal_loc_logits(model, model_name, hparams, tok, [''], loc_a, loc_q, None)
            post_text_loc_logits = icl_multimodal_loc_logits(model, model_name, hparams, tok, icl_examples, loc_a, f'New Fact: {prompt} {target}\nPrompt: {loc_q}', None)

            ret['loc_acc_prompt']  = record["locality_prompt"]
            ret['loc_acc_ground_truth']  = record["locality_ground_truth"]
            ret['loc_acc_predict'] = icl_multimodal_decode(model, model_name, hparams, tok, icl_examples, loc_a, f'New Fact: {prompt} {target}\nPrompt: {loc_q}', None)

            pre_text_loc_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(pre_text_loc_logits.float(), dim=-1), k=1, dim=-1).indices
            post_text_loc_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_text_loc_logits.float(), dim=-1), k=1, dim=-1).indices

            locality_acc = sum(post_text_loc_logits_softmax_top_k.view(-1) == pre_text_loc_logits_softmax_top_k.view(-1))/post_text_loc_logits_softmax_top_k.view(-1).shape[0]

            ret['locality_acc'] = locality_acc
        
        if "multimodal_locality_image" in record.keys():
            pre_image_loc_logits = icl_multimodal_loc_logits(model, model_name, hparams, tok, [''], m_loc_a, m_loc_q, m_loc_image)
            post_image_loc_logits = icl_multimodal_loc_logits(model, model_name, hparams, tok, icl_examples, m_loc_a, f'New Fact: {prompt} {target}\nPrompt: {m_loc_q}', m_loc_image)

            ret['mm_loc_acc_prompt']  = record["multimodal_locality_prompt"]
            ret['mm_loc_acc_ground_truth']  = record["multimodal_locality_ground_truth"]
            ret['mm_loc_acc_predict'] = icl_multimodal_decode(model, model_name, hparams, tok, icl_examples, m_loc_a, f'New Fact: {prompt} {target}\nPrompt: {m_loc_q}', m_loc_image)

            pre_image_loc_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(pre_image_loc_logits.float(), dim=-1), k=10, dim=-1).indices
            post_image_loc_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_image_loc_logits.float(), dim=-1), k=10, dim=-1).indices

            locality_image_acc = sum(post_image_loc_logits_softmax_top_k.view(-1) == pre_image_loc_logits_softmax_top_k.view(-1))/post_image_loc_logits_softmax_top_k.view(-1).shape[0]

            ret['locality_image_acc'] = locality_image_acc
    ###################################################################

        if knowledge_type ==0 or knowledge_type ==1:
            if knowledge_type ==0:
                ret['knowledge_type'] = 0
            elif knowledge_type ==1:
                ret['knowledge_type'] = 1
       
            new_fact_rel_prompt_1 = f'New Fact: {prompt} {target}\nPrompt: {rel_prompt_1}'
            rel_prompt_1_acc, _  = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,rel_ground_truth_1, new_fact_rel_prompt_1, None)
            ret['rel_prompt_1_acc'] = rel_prompt_1_acc
            
            ret['rel_1_acc_prompt'] = record["rel_prompt_1"]
            ret['rel_1_acc_ground_truth'] = record["rel_ground_truth_1"]
            ret['rel_1_acc_predict'] = icl_multimodal_decode(model, model_name, hparams, tok, icl_examples, rel_ground_truth_1, new_fact_rel_prompt_1, None)
        
            new_fact_rel_prompt_2  = f'New Fact: {prompt} {target}\nPrompt: {rel_prompt_2}'
            rel_prompt_2_acc, _ = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,rel_ground_truth_2, new_fact_rel_prompt_2, None)  
            ret['rel_prompt_2_acc'] = rel_prompt_2_acc

            ret['rel_2_acc_prompt'] = record["rel_prompt_2"]
            ret['rel_2_acc_ground_truth']  = record["rel_ground_truth_2"]
            ret['rel_2_acc_predict'] = icl_multimodal_decode(model, model_name, hparams, tok, icl_examples, rel_ground_truth_2, new_fact_rel_prompt_2, None)

        #新增mm reliability+ origin image
        
            new_fact_m_rel_prompt_1_image = f'New Fact: {prompt} {target}\nPrompt: {m_rel_prompt_1}'
            m_rel_prompt_1_image_acc, _ = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,m_rel_ground_truth_1, new_fact_m_rel_prompt_1_image, image)
            ret['m_rel_prompt_1_image_acc'] = m_rel_prompt_1_image_acc

            ret['m_rel_1_acc_prompt'] = record["m_rel_prompt_1"]
            ret['m_rel_1_acc_ground_truth'] = record["m_rel_ground_truth_1"]
            ret['m_rel_1_acc_predict']  = icl_multimodal_decode(model, model_name, hparams, tok, icl_examples, m_rel_ground_truth_1, new_fact_m_rel_prompt_1_image, image)

            new_fact_m_rel_prompt_2_image = f'New Fact: {prompt} {target}\nPrompt: {m_rel_prompt_2}'
            m_rel_prompt_2_image_acc, _ = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,m_rel_ground_truth_2, new_fact_m_rel_prompt_2_image, image)
            ret['m_rel_prompt_2_image_acc'] = m_rel_prompt_2_image_acc

            ret['m_rel_2_acc_prompt'] = record["m_rel_prompt_2"]
            ret['m_rel_2_acc_ground_truth']   = record["m_rel_ground_truth_2"]
            ret['m_rel_2_acc_predict']  = icl_multimodal_decode(model, model_name, hparams, tok, icl_examples, m_rel_ground_truth_2, new_fact_m_rel_prompt_2_image, image)
            
    #新增mm reliability+ rephrase image
        
            #######  m_rel_prompt_1_image_acc
            new_fact_m_rel_prompt_1_image = f'New Fact: {prompt} {target}\nPrompt: {m_rel_prompt_1}'
            m_rel_prompt_1_image_rephrase_acc, _ = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples, m_rel_ground_truth_1, new_fact_m_rel_prompt_1_image, rephrase_image)
            ret['m_rel_prompt_1_image_rephrase_acc'] = m_rel_prompt_1_image_rephrase_acc

            ret['m_rel_1_image_rephrase_acc_prompt'] = record["m_rel_prompt_1"]
            ret['m_rel_1_image_rephrase_acc_ground_truth'] = record["m_rel_ground_truth_1"]
            ret['m_rel_1_image_rephrase_acc_predict'] =  icl_multimodal_decode(model, model_name, hparams, tok, icl_examples, m_rel_ground_truth_1, new_fact_m_rel_prompt_1_image, rephrase_image)

            #######  m_rel_prompt_2_image_acc
            new_fact_m_rel_prompt_2_image = f'New Fact: {prompt} {target}\nPrompt: {m_rel_prompt_2}'
            m_rel_prompt_2_image_rephrase_acc, _ = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples, m_rel_ground_truth_2, new_fact_m_rel_prompt_2_image, rephrase_image)
            ret['m_rel_prompt_2_image_rephrase_acc'] = m_rel_prompt_2_image_rephrase_acc

            ret['m_rel_2_image_rephrase_acc_prompt']  = record["m_rel_prompt_2"]
            ret['m_rel_2_image_rephrase_acc_ground_truth'] = record["m_rel_ground_truth_2"]
            ret['m_rel_2_image_rephrase_acc_predict'] = icl_multimodal_decode(model, model_name, hparams, tok, icl_examples, m_rel_ground_truth_2, new_fact_m_rel_prompt_2_image, rephrase_image)

        elif knowledge_type ==2:
            
            ret['knowledge_type'] = 2

            #新增text reliability  引入def compute_icl_edit_quality()中的计算格式，利用def icl_lm_eval()函数 用于文本问题计算
            new_fact_rel_prompt = f'New Fact: {prompt} {target}\nPrompt: {rel_prompt}'
            rel_prompt_acc, _  = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,rel_ground_truth, new_fact_rel_prompt, None)
            ret['rel_prompt_acc'] = rel_prompt_acc


            ret['rel_acc_prompt']  = record["rel_prompt"]
            ret['rel_acc_ground_truth'] = record["rel_ground_truth"]
            ret['rel_acc_predict'] = icl_multimodal_decode(model, model_name, hparams, tok, icl_examples, rel_ground_truth, new_fact_rel_prompt, None)

        #新增mm reliability+ origin image

            new_fact_m_rel_prompt_image = f'New Fact: {prompt} {target}\nPrompt: {m_rel_prompt}'
            m_rel_prompt_image_acc, _ = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,m_rel_ground_truth, new_fact_m_rel_prompt_image, image)
            ret['m_rel_prompt_image_acc'] = m_rel_prompt_image_acc

            ret['m_rel_acc_prompt'] = record["m_rel_prompt"]
            ret['m_rel_acc_ground_truth'] = record["m_rel_ground_truth"]
            ret['m_rel_acc_predict'] = icl_multimodal_decode(model, model_name, hparams, tok, icl_examples, m_rel_ground_truth, new_fact_m_rel_prompt_image, image)


            #######  m_rel_prompt_1_image_acc
            new_fact_m_rel_prompt_image = f'New Fact: {prompt} {target}\nPrompt: {image_rephrase_question}'
            m_rel_prompt_image_rephrase_acc, _ = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples, m_rel_ground_truth, new_fact_m_rel_prompt_image, rephrase_image)
            ret['m_rel_prompt_image_rephrase_acc'] = m_rel_prompt_image_rephrase_acc

            ret['m_rel_image_rephrase_acc_prompt']  = record["m_rel_prompt"]
            ret['m_rel_image_rephrase_acc_ground_truth'] = record["m_rel_ground_truth"]
            ret['m_rel_image_rephrase_acc_predict'] = icl_multimodal_decode(model, model_name, hparams, tok, icl_examples, m_rel_ground_truth, new_fact_m_rel_prompt_image, rephrase_image)


    ######### portability #########
    if pre_edit:
        ret['portability_acc'] = None
    else:
        if "portability_prompt" in record.keys():
            assert len(record['portability_prompt'])==1, "Portability evaluation only has one prompt at a time"
            port_acc = 0
            if knowledge_type ==0 or knowledge_type ==1:
                for port_q, port_a in zip(record['portability_prompt'], record['portability_ground_truth']):
                    port_acc_i, pred_targ_ids = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples, port_a, f'New Fact: {prompt} {target}\nPrompt: {port_q}', image)
                    port_acc += port_acc_i
                ret['portability_acc'] = port_acc/len(record['portability_prompt'])
                ret['pred_ids'] = pred_targ_ids[0].tolist()
                ret['targ_ids'] = pred_targ_ids[1].tolist()

                ret['port_prompt'] = record["portability_prompt"]
                ret['port_ground_truth'] = record["portability_ground_truth"]
                ret['port_predict'] = tok.decode(pred_targ_ids[0][0], skip_special_tokens=True)

            elif knowledge_type ==2:
                for port_q, port_a in zip(record['portability_prompt'], record['portability_ground_truth']):
                    port_acc_i, pred_targ_ids = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples, port_a, f'New Fact: {prompt} {target}\nPrompt: {port_q}', one_hop_img)
                    port_acc += port_acc_i
                ret['portability_acc'] = port_acc/len(record['portability_prompt'])
                ret['pred_ids'] = pred_targ_ids[0].tolist()
                ret['targ_ids'] = pred_targ_ids[1].tolist()
                ret['port_prompt'] = record["portability_prompt"]
                ret['port_ground_truth'] = record["portability_ground_truth"]
                ret['port_predict'] = tok.decode(pred_targ_ids[0][0], skip_special_tokens=True)
    # 打开文件并追加内容
    import json
    import os
    # 将 Tensor 转换为列表的递归函数
    def tensor_to_list(obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()  # 将 Tensor 转换为列表
        elif isinstance(obj, dict):
            return {k: tensor_to_list(v) for k, v in obj.items()}  # 递归处理 dict
        elif isinstance(obj, list):
            return [tensor_to_list(i) for i in obj]  # 递归处理 list
        return obj  # 如果不是 Tensor，就直接返回

# 将包含 Tensor 的 dict 转换为 JSON 可序列化的格式
    ret_serializable = tensor_to_list(ret)
    from datetime import datetime
    
    # if hparams.data_type == 'entity':
    #     hparams.results_dir = "./results/entity"
    # elif hparams.data_type == 'visual':
    #     hparams.results_dir = "./results/visual"
    # elif hparams.data_type == 'user':
    #     hparams.results_dir = "./results/user"

    hparams.results_dir = f"./results/{hparams.data_type}"

    # 固定文件名，而不是每次使用当前时间
    model_dir = os.path.join(hparams.results_dir, "IKE")
    decode_path = f'{model_dir}/{hparams.model_name}_IKE_k={hparams.k}.json'

    # 检查文件夹是否存在，不存在则创建
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # 检查文件是否存在并读取现有的内容
    if os.path.exists(decode_path):
        with open(decode_path, 'r') as json_file:
            try:
                data = json.load(json_file)  # 尝试读取现有的JSON内容
            except json.JSONDecodeError:
                data = []  # 如果文件是空的或无法解析，使用空列表
    else:
        data = []  # 文件不存在时，使用空列表

    # 追加新的数据到列表中
    data.append(ret_serializable)

    # 将更新后的列表写入文件中
    with open(decode_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)  # 以4个空格缩进写入JSON内容
        json_file.write("\n")  # 可选，写入换行符以保持格式整齐
    ######### portability #########

    return ret

def icl_multimodal_lm_eval(
        model,
        model_name,
        hparams: HyperParams,
        tokenizer,
        icl_examples,
        target,
        x,
        image,
        neighborhood=False
)-> typing.Dict:
    device = torch.device(f'cuda:{hparams.device}')
    
    samples = prepare_multimodal_edit(hparams, tokenizer, target, [''.join(icl_examples) + f'{x}'], image) 
    
    return compute_multimodal_edit_quality(model, samples)

def icl_multimodal_loc_logits(
        model,
        model_name,
        hparams: HyperParams,
        tokenizer,
        icl_examples,
        target,
        x,
        image,
        neighborhood=False
)-> typing.Dict:    
    batch = prepare_multimodal_edit(hparams, tokenizer, target, [''.join(icl_examples) + f'{x}'], image) 
    
    with torch.no_grad():
        if "qwen" in model.__class__.__name__.lower():
            outputs = model(batch['inputs'].to(hparams.device))
        elif "owl" in model.__class__.__name__.lower():
            input_ids, image = batch['input_ids'], batch['image']
            # from torch.cuda.amp import autocast
            # with autocast():
            outputs = model(input_ids.to(hparams.device), 
                                        images=image.to(hparams.device, dtype=torch.float16))
        else:
            outputs = model(batch)
        if isinstance(outputs, torch.Tensor):
            logits = outputs.detach().cpu()
        else:
            logits = outputs.logits.detach().cpu()    
        # targ = outputs.labels.detach().cpu()
        targ = batch["labels"].cpu()
    if logits.dim() == 3:
        logits = logits[:, :-1]
        # targ = targ[:, 1:]
        logits = logits[:, -targ.shape[1]:]
    else:
        raise ValueError("logits should have 3 dimensions")
    return logits

def icl_multimodal_decode(
        model,
        model_name,
        hparams: HyperParams,
        tokenizer,
        icl_examples,
        target,
        x,
        image,
        neighborhood=False
)-> typing.Dict:    
    batch = prepare_multimodal_edit(hparams, tokenizer, target, [''.join(icl_examples) + f'{x}'], image) 
    
    with torch.no_grad():
        if "qwen" in model.__class__.__name__.lower():
            outputs = model(batch['inputs'].to(hparams.device))
        elif "owl" in model.__class__.__name__.lower():
            input_ids, image = batch['input_ids'], batch['image']
            # from torch.cuda.amp import autocast
            # with autocast():
            outputs = model(input_ids.to(hparams.device), 
                                        images=image.to(hparams.device, dtype=torch.float16))
        else:
            outputs = model(batch)
        if isinstance(outputs, torch.Tensor):
            logits = outputs.detach().cpu()
        else:
            logits = outputs.logits.detach().cpu()    
        # targ = outputs.labels.detach().cpu()
        targ = batch["labels"].cpu()
    if logits.dim() == 3:
        logits = logits[:, :-1]
        # targ = targ[:, 1:]
        logits = logits[:, -targ.shape[1]:]
    else:
        raise ValueError("logits should have 3 dimensions")
        
    # 创建掩码并处理标签
    mask = targ != -100
    targ[~mask] = 0
    # 获取预测ID并根据掩码填充
    pred_ids = logits.argmax(-1).masked_fill(~mask, 0).detach().cpu()
    
    # 解码预测ID为字符串
    predict = tokenizer.decode(pred_ids.tolist()[0], skip_special_tokens=True)
    return predict

def prepare_multimodal_edit(hparams,
                            tok,
                            target,
                            prompts,
                            image):
    if isinstance(target, str):
        target = [target,]
    if isinstance(prompts, str):
        prompts = [prompts,]
    if "qwen-vl" in hparams.model_name.lower():
        ret = {
            'inputs': tok(f'{prompts[0]} {target[0]}', return_tensors='pt')["input_ids"] if image is None else tok(f'Picture 1: <img>{image}</img>\n{prompts[0]} {target[0]}', return_tensors='pt')["input_ids"],
            'labels': tok(" " + target[0], add_special_tokens=False, return_tensors="pt",)["input_ids"],
        }

    elif 'owl' in hparams.model_name.lower():
        from ..trainer.mPLUG_Owl2.mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from ..trainer.mPLUG_Owl2.mplug_owl2.mm_utils import tokenizer_image_token
        prompt = prompts[0] + " " + target[0]
        if image is not None:
            prompt = DEFAULT_IMAGE_TOKEN + prompt
        else:
            image = torch.zeros(1, 3, 448, 448)
        ret = {
            'input_ids': tokenizer_image_token(prompt, tok, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0),
            'labels': tok(target, add_special_tokens=False, return_tensors="pt",)["input_ids"],
            'image': image
        }

    else:
        if torch.is_tensor(image) and image is not None and len(image.shape) == 3:
            image = image.unsqueeze(0)
        text_input = [prompt_ + ' ' + target_ for prompt_, target_ in zip(prompts, target)]
        prompts_len = [len(tok.encode(prompt, add_special_tokens=False)) for prompt in prompts]
        target = tok(target, add_special_tokens=False, return_tensors="pt",)["input_ids"]
            
        ret = {
            'text_input': text_input,
            'image': image,
            'labels': target,
            'prompts_len': prompts_len        
        } 
    return ret

def compute_multimodal_edit_quality(model, batch):
    
    with torch.no_grad():
        if "qwen" in model._get_name().lower():
            outputs = model(batch['inputs'].to(model.device))
        elif "owl" in model._get_name().lower():
            input_ids, image = batch['input_ids'], batch['image']
            outputs = model(input_ids.to(model.device), 
                                         images=image.to(model.device, dtype=torch.float16))
        else:     
            outputs = model(batch)
        if isinstance(outputs, torch.Tensor):
            logits = outputs.detach().cpu()
        else:
            logits = outputs.logits.detach().cpu()    
        # targ = outputs.labels.detach().cpu()
        targ = batch["labels"].cpu()
    if logits.dim() == 3:
        logits = logits[:, :-1]
        # targ = targ[:, 1:]
        logits = logits[:, -targ.shape[1]:]
    mask = targ != -100
    targ[~mask] = 0
    pred_ids = logits.argmax(-1).masked_fill(~mask, 0).detach().cpu()
    correct = pred_ids == targ
    correct = correct & mask
    num_non_padding = mask.sum().float().item()
    acc = correct.sum() / num_non_padding
    
    return acc, (pred_ids.numpy(), targ.numpy())
  
def compute_multimodal_edit_quality_demo(model, batch):
    
    with torch.no_grad():
        outputs = model(batch)
        if isinstance(outputs, torch.Tensor):
            logits = outputs.detach().cpu()
        else:
            logits = outputs.logits.detach().cpu()    
        # targ = outputs.labels.detach().cpu()
        targ = batch["labels"].cpu()
    if logits.dim() == 3:
        logits = logits[:, :-1]
        # targ = targ[:, 1:]
        logits = logits[:, -targ.shape[1]:]
    mask = targ != -100
    targ[~mask] = 0
    pred_ids = logits.argmax(-1).masked_fill(~mask, 0).detach().cpu()
    correct = pred_ids == targ
    correct = correct & mask
    num_non_padding = mask.sum().float().item()
    acc = correct.sum() / num_non_padding
    
    return acc, pred_ids.numpy(), logits

def compute_multimodal_edit_results(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    record: typing.Dict,
    device
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :paran snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """
    ret = {}
    # First, unpack rewrite evaluation record.
    
    target = record["target"]
    rewrite_prompts = record["prompt"]
    image = record["image"]
    
    edit_inner = prepare_multimodal_edit(hparams, tok, target, rewrite_prompts, image)
    ret['rewrite_acc'], _ = compute_multimodal_edit_quality(model, edit_inner)
    
    if "rephrase_prompt" in record.keys():
        rephrase_prompts = record["rephrase_prompt"]
        edit_outer = prepare_multimodal_edit(hparams, tok, target, rephrase_prompts, image)
        ret['rephrase_acc'], _ = compute_multimodal_edit_quality(model, edit_outer)
        
    if "image_rephrase" in record.keys():
        rephrase_image = record["image_rephrase"]
        edit_image_outer = prepare_multimodal_edit(hparams, tok, target, rewrite_prompts, rephrase_image) 
        ret['image_rephrase_acc'], _ = compute_multimodal_edit_quality(model, edit_image_outer)

    if 'locality_prompt' in record.keys():
        locality_prompt = record["locality_prompt"]
        locality_ground_truth = record["locality_ground_truth"]
        locality = prepare_multimodal_edit(hparams, tok, locality_ground_truth, locality_prompt, None)
        _, ret['locality_output'] = compute_multimodal_edit_quality(model, locality)
        
    if 'multimodal_locality_prompt' in record.keys():
        m_loc_prompt = record["multimodal_locality_prompt"]
        m_loc_ground_truth = record["multimodal_locality_ground_truth"]
        m_loc_image = record["multimodal_locality_image"]
        m_locality = prepare_multimodal_edit(hparams, tok, m_loc_ground_truth, m_loc_prompt, m_loc_image)
        _, ret['multimodal_locality_output'] = compute_multimodal_edit_quality(model, m_locality)
    # Form a list of lists of prefixes to test.

    return ret
  
def compute_multimodal_edit_results_demo(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    record: typing.Dict,
    device
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :paran snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """
    ret = {}
    # First, unpack rewrite evaluation record.
    
    target = record["target"]
    rewrite_prompts = record["prompt"]
    image = record["image"]
    
    edit_inner = prepare_multimodal_edit(hparams, tok, target, rewrite_prompts, image)
    ret['rewrite_acc'], _, logits = compute_multimodal_edit_quality_demo(model, edit_inner)
    
    if "rephrase_prompt" in record.keys():
        rephrase_prompts = record["rephrase_prompt"]
        edit_outer = prepare_multimodal_edit(hparams, tok, target, rephrase_prompts, image)
        ret['rephrase_acc'], _ = compute_multimodal_edit_quality(model, edit_outer)
        
    if "image_rephrase" in record.keys():
        rephrase_image = record["image_rephrase"]
        edit_image_outer = prepare_multimodal_edit(hparams, tok, target, rewrite_prompts, rephrase_image) 
        ret['image_rephrase_acc'], _ = compute_multimodal_edit_quality(model, edit_image_outer)

    if 'locality_prompt' in record.keys():
        locality_prompt = record["locality_prompt"]
        locality_ground_truth = record["locality_ground_truth"]
        locality = prepare_multimodal_edit(hparams, tok, locality_ground_truth, locality_prompt, None)
        _, ret['locality_output'] = compute_multimodal_edit_quality(model, locality)
        
    if 'multimodal_locality_prompt' in record.keys():
        m_loc_prompt = record["multimodal_locality_prompt"]
        m_loc_ground_truth = record["multimodal_locality_ground_truth"]
        m_loc_image = record["multimodal_locality_image"]
        m_locality = prepare_multimodal_edit(hparams, tok, m_loc_ground_truth, m_loc_prompt, m_loc_image)
        _, ret['multimodal_locality_output'] = compute_multimodal_edit_quality(model, m_locality)
    # Form a list of lists of prefixes to test.

    return ret, logits


    prompt_tok = tok(
        prompt,
        padding=True,
        truncation=True,
        max_length=hparams.max_length,
        return_tensors="pt",
    ).to(f"cuda:{device}")

    trg_tok = tok(
        target,
        padding=True,
        truncation=True,
        max_length=hparams.max_length,
        return_tensors="pt",
    ).to(f"cuda:{device}")

    prompt_tok['labels'] = trg_tok['input_ids']
    # prompt_tok['decoder_attention_mask'] = trg_tok['attention_mask']


    with torch.no_grad():
        outputs = model(**prompt_tok)
        if type(outputs) is torch.Tensor:
            logits = outputs
        else:
            logits = outputs.logits

        assert logits.size(1) == trg_tok['input_ids'].size(1)
        ans = torch.argmax(logits, dim=-1)
        if locality:
            return ans.squeeze().detach().cpu().numpy().tolist()

        return torch.mean((trg_tok['input_ids'][:,:-1] == ans[:,:-1]).float(), dim=-1).detach().cpu().numpy().tolist()[0]

def compute_sent_metric(
    model,
    edited_model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    metric_kwargs: typing.Dict,
    device,
    test_generation=True
    ):
    
    if "llama" not in model_name:
        raise NotImplementedError("currently only support for llama")
        
    def get_edit_labels(ids, prompts=None):
        labels = ids.clone()
        labels[labels == tok.pad_token_id] = -100
        return labels
        
    same_mask = torch.tensor([i == o for i, o in zip(metric_kwargs["inner_target"], metric_kwargs["all_target"])], device=device)
    edit_toks = {
        f"{k1}_{k2}": v2.to(device)
        for k1, v1 in {
            "inner": metric_kwargs["inner_all_qa"],
            "outer": metric_kwargs["outer_all_qa"],
        }.items()
        for k2, v2 in tok(
            v1,
            return_tensors="pt",
            padding=True,
            max_length=128,
            truncation=True,
        ).items()
    }
    for key in ["inner", "outer"]:
        value = edit_toks[f"{key}_input_ids"]
        mask = [([True] * value.shape[-1])] * value.shape[0]
        for i in range(value.shape[0]):
            sep_idx = list(value[i]).index(tok.convert_tokens_to_ids("</s>"))
            for j in range(sep_idx): #连带</s>一块mask掉
                mask[i][j] = False
        edit_toks[key + "_q_mask"] = torch.tensor(mask).to(device)

    with torch.no_grad():
        inner_base_logits = model(
            input_ids=edit_toks["inner_input_ids"],
            attention_mask=edit_toks["inner_attention_mask"],   
        )["logits"]
        inner_edit_logits = edited_model(
            input_ids=edit_toks["inner_input_ids"],
            attention_mask=edit_toks["inner_attention_mask"],   
        )["logits"]
        
        outer_base_logits = model(
            input_ids=edit_toks["outer_input_ids"],
            attention_mask=edit_toks["outer_attention_mask"],   
        )["logits"]
        outer_edit_logits = edited_model(
            input_ids=edit_toks["outer_input_ids"],
            attention_mask=edit_toks["outer_attention_mask"],   
        )["logits"]
    
    result = {
        "es": es_sent(inner_base_logits, inner_edit_logits, edit_toks["inner_q_mask"], get_edit_labels(edit_toks["inner_input_ids"]), same_mask).item(),
        "dd": kl_loc_loss(outer_base_logits, outer_edit_logits, edit_toks["outer_q_mask"]).item(),
    }
    if  test_generation:
        result['fluency'] = test_generation_quality(model=model,tok=tok,prefixes=metric_kwargs["inner_q"] if isinstance(metric_kwargs["inner_q"],list) else [metric_kwargs["inner_q"],], max_out_len=100)
    return result
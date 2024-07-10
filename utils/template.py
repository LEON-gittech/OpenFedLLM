"""
To support TRL supervised fine-tuning. Right now, we need to manually set the template here.
"""

alpaca_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{} 

### Response: {}{}""" #（instruction，response，eos）

vicuna_template = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {} ASSISTANT: {}{}"""

TEMPLATE_DICT = {
    'alpaca': (alpaca_template, '\n### Response:'),
    'vicuna': (vicuna_template, ' ASSISTANT:'),
}


def get_formatting_prompts_func(template_name, eos_token):
    overall_temp, response_temp = TEMPLATE_DICT[template_name]
    def formatting_prompts_func(example): #这个闭包有点东西的
        output_texts = []    
        # print("instruction", len(example["instruction"]))
        # print("response", len(example["response"]))
        for i in range(len(example['instruction'])):    
            text = overall_temp.format(example['instruction'][i], example['response'][i], eos_token)    
            output_texts.append(text)    
        # print(len(output_texts))
        return output_texts    
    return formatting_prompts_func, response_temp

# def get_formatting_prompts_func(template_name, eos_token):
#     return '\n### Response:'

# def formatting_prompts_func(example): #这个闭包有点东西的
#     output_texts = []    
#     print("instruction", len(example["instruction"]))
#     print("response", len(example["response"]))
#     for i in range(len(example['instruction'])):    
#         text = alpaca_template.format(example['instruction'][i], example['response'][i], "<|end_of_text|>")    
#         output_texts.append(text)    
#     return output_texts    

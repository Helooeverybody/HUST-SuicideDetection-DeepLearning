from transformers import BertForSequenceClassification
from transformers import BertTokenizer,BertConfig
import torch 
import math


def answer(query : str) -> str:
    # load model 
    config = BertConfig.from_pretrained("UI/utils/suicide_pretrained", num_hidden_layers=1,num_labels=2)
    model = BertForSequenceClassification.from_pretrained("UI/utils/suicide_pretrained",config = config)

    # load tokenizer
    tokenizer = BertTokenizer.from_pretrained("UI/utils/suicide_pretrained")
    input_token = tokenizer(query, truncation = True, padding = True)

    # predict
    labels = {0:'non-suicide' , 1:'suicide'}
    model.eval()
    output = model(torch.tensor([input_token['input_ids']]))
    pred = torch.argmax(output.logits, dim = 1).item()
    x=output.logits[0,1].item()
    p=math.exp(x)/(1+math.exp(x))
    ans=""
    if 0<=p<=0.4:
        ans="Thank you for reaching out. It seems like things might be okay, but if you're ever in need of support or someone to talk to, we're here for you."
    elif 0.4<p<=0.75:
        ans="Thank you for sharing with us. If things are feeling a bit tough, remember you're not alone. Let us know if you'd like to talk more."
    else:
        ans="It sounds like you’re very depressed. If it’s okay with you, let’s discuss ways we can offer more support. Your well-being matters to us."
    return ans

if __name__ == '__main__':
    result = answer(query = ''' Things have been really tough lately, and sometimes I just feel completely overwhelmed. It’s like I’m constantly running and getting nowhere, trying to keep up with everything but feeling like I’m falling behind. I know life has its ups and downs, and maybe this is just one of those tough times. I keep reminding myself that things will get better, even if it doesn’t feel that way now. I just need to take it one day at a time, keep pushing through, and maybe find little things each day that can make me feel a bit better''')
    
    print(result)
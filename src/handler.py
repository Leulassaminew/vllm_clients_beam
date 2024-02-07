import runpod
from utils import JobInput
from engine import vLLMEngine
from transformers import pipeline,BertForSequenceClassification,BertTokenizer
import torch

vllm_engine = vLLMEngine()
def load_model():
    global model,tokenize
    if not model:
        tokenize = BertTokenizer.from_pretrained('meetplace1/bertclassify900',
                                                      do_lower_case=True)
        model = BertForSequenceClassification.from_pretrained("meetplace1/bertclassify900",
                                                                  num_labels=15,
                                                                  output_attentions=False,
                                                                  output_hidden_states=False)
    return model,tokenize
    
model=None
tokenize=None

async def handler(job):
    j=job["input"]
    t=j["task"]
    if t!="report":
        messages=j["messages"][-1]
        messages=messages["content"]
        count_usage=j.pop("count_usage")
        score=j.pop("score")
        model,tokenize=load_model()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        encoded_input = tokenize(messages, return_tensors='pt')
        encoded_input = encoded_input.to(device)
        model=model.to(device)
        output = model(**encoded_input)
        d = output.logits[0].to('cpu').detach().numpy()
        ind=[]
        if d[14]>0.5:
            ind=[14]
        else:
            for i in range(15):
                if d[i]>4:
                    ind.append(i)
        if 14 in ind:
            if score>0:
                score-=1
            else:
                score=0
            count_usage[14]+=1
        else:
            for item in ind:
                if count_usage[item]==0:
                    score+=10
                    count_usage[item]+=1
                else:
                    score+=1
        j["score"]=score
        j["count_usage"]=count_usage
    job_input = JobInput(j)
    results_generator = vllm_engine.generate(job_input)
    async for batch in results_generator:
        yield batch

runpod.serverless.start(
    {
        "handler": handler,
        "concurrency_modifier": lambda x: vllm_engine.max_concurrency,
        "return_aggregate_stream": True,
    }
)
       

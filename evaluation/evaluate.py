from argparse import ArgumentParser
from sense.sense_m import Agent
import pandas as pd
from tqdm import tqdm
import json

def main(model, dataset,index=None):
    
    df = pd.read_csv(dataset)
    fname = f"./outputs/{model}_results.json"
    # index = ""
    current_index = ""
    results = []
    if index is not None:
        print(index)
        df = df[df['INDEX NSX'] == index]
        print(df.info())
        
    data = df.to_dict(orient="records")
    for row in tqdm(data):
        if row['INDEX NSX'] != current_index:
            current_index = row['INDEX NSX']
            agent = Agent(model=model, index=current_index,edital=f"FUNDEP {row['SIGLA MENU']}",logs_file="logs/logs_gpt4.jsonl")
        answer = agent(row['PERGUNTA'])
        results.append({
            "question": row['PERGUNTA'],
            "reference": row['RESPOSTA ESPERADA'],
            "predicted": answer,
        })
        json.dump(results,open(fname,"w"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-0613")
    parser.add_argument("--dataset", type=str, default="./data/dataset.csv")
    parser.add_argument("--index", type=str, default="FUNDEP_Paraopeba")

    args = parser.parse_args()
    main(**vars(args))
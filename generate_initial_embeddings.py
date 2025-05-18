import torch
import os
from transformers import LlamaModel, AutoTokenizer, AutoConfig
import json 

def load_vocab(ori_dataset_path):
    rel2id_file = os.path.join(ori_dataset_path, "relation2id.json")
    ent2id_file = os.path.join(ori_dataset_path, "entity2id.json")
    relation2id_old = json.load(open(rel2id_file))
    relation2id_old = dict((k.lower(), v) for k, v in relation2id_old.items())
    relation2id = relation2id_old.copy()
    counter = len(relation2id_old)
    for relation in relation2id_old:
        relation2id["Inv_" + relation] = counter
        counter += 1
    id2relation = dict([(v, k) for k, v in relation2id.items()])
    ent2id = json.load(open(ent2id_file))
    ent2id = dict((k.lower(), v) for k, v in ent2id.items())
    id2ent = dict([(v, k) for k, v in ent2id.items()])
    return {"rel2id": relation2id, "id2rel": id2relation, "ent2id": ent2id, "id2ent": id2ent}

def create_and_save_embeddings(dataset_name, hf_model_name, device="cuda:0", ori_data_path="./data/original"):
    print(f"Attempting to generate initial embeddings for {dataset_name} using {hf_model_name}")
    vocab_path = os.path.join(ori_data_path, dataset_name)
    vocab_dict = load_vocab(vocab_path)
    ent2id = vocab_dict["ent2id"]
    rel2id = vocab_dict["rel2id"]

    all_ent = list(ent2id.keys()) + ['PAD']
    all_rel = list(rel2id.keys()) + ['PAD']

    embedding_file_path = os.path.join(ori_data_path, dataset_name, 'ent_rel_emb.pt')

    if os.path.exists(embedding_file_path):
        print(f"Embedding file {embedding_file_path} already exists. Skipping generation.")
        return

    try:
        print(f"Loading tokenizer for {hf_model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token 
        print(f"Loading LlamaModel {hf_model_name} onto {device} for embedding generation...")
        emb_model = LlamaModel.from_pretrained(hf_model_name, trust_remote_code=True).to(device)
        emb_model.eval() 

        with torch.no_grad():
            print("Tokenizing entities and relations...")
            input_ents = tokenizer(all_ent, return_tensors='pt', padding="longest", truncation=True, max_length=64).input_ids.to(device)
            input_rels = tokenizer(all_rel, return_tensors='pt', padding="longest", truncation=True, max_length=64).input_ids.to(device)

            print("Generating embeddings...")
            embedding_layer = emb_model.embed_tokens
            ent_embedding = embedding_layer(input_ents).mean(dim=1)
            rel_embedding = embedding_layer(input_rels).mean(dim=1)

            output_dir = os.path.dirname(embedding_file_path)
            os.makedirs(output_dir, exist_ok=True)
            torch.save(torch.cat([ent_embedding, rel_embedding], dim=0), embedding_file_path)
            print(f"Generated and saved embedding file to {embedding_file_path}")
    except Exception as e:
        print(f"Error during embedding generation: {e}")
    finally:
        if 'emb_model' in locals():
            del emb_model
        torch.cuda.empty_cache()

if __name__ == "__main__":

    create_and_save_embeddings(dataset_name="icews14", hf_model_name="meta-llama/Llama-2-7b-hf")
    create_and_save_embeddings(dataset_name="icews18", hf_model_name="meta-llama/Llama-2-7b-hf")
    create_and_save_embeddings(dataset_name="GDELT", hf_model_name="meta-llama/Llama-2-7b-hf")
    print("Initial embedding generation process finished.")
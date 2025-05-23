import json
import os

def load_json_file(file_path):
    """JSON 파일을 로드합니다."""
    print(f"Loading: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: File not found at {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_id_to_name_mapping(name_to_id_dict):
    """name_to_id 딕셔너리로부터 id_to_name 딕셔너리를 생성합니다."""
    return {v: k for k, v in name_to_id_dict.items()}

def normalize_relation_name(name, for_system):
    name_lower = name.lower()
    if name_lower.startswith("inv_") or name_lower.startswith("inv "): # LLM-DA 스타일
        base_name = name_lower.replace("inv_", "").replace("inv ", "").strip()
        return f"inv_{base_name}"
    elif name_lower.startswith("inv_") and for_system == "spark": # SPARK 스타일 (만약 다르다면)
        base_name = name_lower.replace("inv_", "").strip() 
        return f"inv_{base_name}" 
    return name_lower # 기본 이름

def convert_rules(
    spark_ent2id_path, spark_rel2id_path,
    llmda_ent2id_path, llmda_rel2id_path,
    llmda_rules_input_path, # 예: confidence.json
    converted_rules_output_path
):
    print("--- 어휘 파일 로딩 시작 ---")
    # 1. SPARK 
    spark_ent2id = load_json_file(spark_ent2id_path)
    spark_rel2id = load_json_file(spark_rel2id_path)
    spark_id2ent = create_id_to_name_mapping(spark_ent2id)
    spark_id2rel = create_id_to_name_mapping(spark_rel2id)
    print("SPARK 어휘 로드 완료.")

    # 2. LLM-DA 
    llmda_ent2id = load_json_file(llmda_ent2id_path)
    llmda_rel2id = load_json_file(llmda_rel2id_path)
    llmda_id2ent = create_id_to_name_mapping(llmda_ent2id)
    llmda_id2rel = create_id_to_name_mapping(llmda_rel2id)
    print("LLM-DA 어휘 로드 완료.")

    print("\n--- ID 매핑 생성 시작 ---")
    # 3. LLM-DA ID -> SPARK ID 매핑 생성 (이름 기준)
    # 엔티티 이름은 동일하다고 가정했으므로, LLM-DA의 ID -> LLM-DA 이름 -> SPARK ID
    llmda_ent_id_to_spark_ent_id = {}
    missing_entities_count = 0
    for llmda_id, llmda_name in llmda_id2ent.items():
        # 엔티티 이름 정규화 (필요시) - 여기서는 단순 소문자화 가정
        # normalized_llmda_name = llmda_name.lower()
        # normalized_spark_ent2id = {k.lower(): v for k, v in spark_ent2id.items()}
        # if normalized_llmda_name in normalized_spark_ent2id:
        #     llmda_ent_id_to_spark_ent_id[llmda_id] = normalized_spark_ent2id[normalized_llmda_name]
        if llmda_name in spark_ent2id: # 이름이 정확히 같다고 가정
            llmda_ent_id_to_spark_ent_id[llmda_id] = spark_ent2id[llmda_name]
        else:
            # print(f"Warning: LLM-DA entity name '{llmda_name}' (ID: {llmda_id}) not found in SPARK's entity vocabulary. Skipping.")
            missing_entities_count +=1
    if missing_entities_count > 0:
        print(f"Warning: {missing_entities_count} LLM-DA entity names were not found in SPARK's entity vocabulary.")

    # 관계 이름은 역관계 접두사 등 주의 필요. 사용자가 이름이 같다고 했으므로,
    # "inv_relation_name" (LLM-DA) 과 "Inv_relation_name" (SPARK) 같은 차이가 없다면 직접 매핑.
    # 만약 차이가 있다면 normalize_relation_name 함수와 유사한 정규화 필요.
    # 여기서는 이름이 정확히 일치한다고 가정 (역관계 포함).
    llmda_rel_id_to_spark_rel_id = {}
    missing_relations_count = 0
    for llmda_id, llmda_name in llmda_id2rel.items():
        # normalized_llmda_name = normalize_relation_name(llmda_name, "llmda")
        # normalized_spark_rel2id = {normalize_relation_name(k, "spark"): v for k, v in spark_rel2id.items()}
        # if normalized_llmda_name in normalized_spark_rel2id:
        #     llmda_rel_id_to_spark_rel_id[llmda_id] = normalized_spark_rel2id[normalized_llmda_name]
        if llmda_name in spark_rel2id: # 이름이 정확히 같다고 가정
             llmda_rel_id_to_spark_rel_id[llmda_id] = spark_rel2id[llmda_name]
        else:
            # print(f"Warning: LLM-DA relation name '{llmda_name}' (ID: {llmda_id}) not found in SPARK's relation vocabulary. Skipping.")
            missing_relations_count +=1
    if missing_relations_count > 0:
        print(f"Warning: {missing_relations_count} LLM-DA relation names were not found in SPARK's relation vocabulary.")
    print("ID 매핑 생성 완료.")

    print("\n--- LLM-DA 규칙 파일 로딩 및 변환 시작 ---")
    # 4. LLM-DA 규칙 파일 로드
    llmda_rules_data = load_json_file(llmda_rules_input_path)
    converted_rules_data = {}
    
    num_rules_processed = 0
    num_rules_skipped_head = 0
    num_rules_partially_skipped_body = 0

    # 5. 규칙 ID 변환
    # llmda_rules_data는 {llmda_head_rel_id_str: [rule_dict_list]} 형태일 것으로 예상 (reasoning.py load_rules 참고)
    for llmda_head_rel_id_str, rules_list in llmda_rules_data.items():
        llmda_head_rel_id = int(llmda_head_rel_id_str) # 키가 문자열 ID일 수 있음
        
        if llmda_head_rel_id not in llmda_rel_id_to_spark_rel_id:
            # print(f"Warning: LLM-DA head relation ID {llmda_head_rel_id} cannot be mapped to SPARK ID. Skipping rules for this head.")
            num_rules_skipped_head += len(rules_list)
            continue
            
        spark_head_rel_id = llmda_rel_id_to_spark_rel_id[llmda_head_rel_id]
        converted_rules_data[str(spark_head_rel_id)] = [] # SPARK ID(문자열 키)로 초기화

        for rule_dict_llmda in rules_list:
            num_rules_processed += 1
            converted_rule = rule_dict_llmda.copy() # 일단 복사 후 ID만 변경

            # head_rel 변환
            if rule_dict_llmda["head_rel"] in llmda_rel_id_to_spark_rel_id:
                converted_rule["head_rel"] = llmda_rel_id_to_spark_rel_id[rule_dict_llmda["head_rel"]]
            else:
                # print(f"Warning: Rule's head_rel ID {rule_dict_llmda['head_rel']} cannot be mapped. Skipping this rule.")
                num_rules_partially_skipped_body +=1
                continue

            # body_rels 변환
            new_body_rels = []
            skip_this_rule_due_to_body = False
            for body_rel_id_llmda in rule_dict_llmda["body_rels"]:
                if body_rel_id_llmda in llmda_rel_id_to_spark_rel_id:
                    new_body_rels.append(llmda_rel_id_to_spark_rel_id[body_rel_id_llmda])
                else:
                    # print(f"Warning: Rule's body_rel ID {body_rel_id_llmda} (in rule for head {rule_dict_llmda['head_rel']}) cannot be mapped. Skipping this rule.")
                    skip_this_rule_due_to_body = True
                    num_rules_partially_skipped_body +=1
                    break
            
            if skip_this_rule_due_to_body:
                continue

            converted_rule["body_rels"] = new_body_rels
            
            # var_constraints는 엔티티 "위치"에 대한 것이므로 ID 변환 불필요.
            # conf, rule_supp, body_supp, llm_confidence 등 다른 필드는 그대로 유지.
            
            converted_rules_data[str(spark_head_rel_id)].append(converted_rule)
    
    print(f"규칙 변환 완료: 총 {num_rules_processed}개 LLM-DA 규칙 처리.")
    if num_rules_skipped_head > 0:
        print(f"  - 헤드 관계 ID 매핑 불가로 건너뛴 규칙 그룹 수 (해당 헤드의 모든 규칙): {num_rules_skipped_head}개 규칙에 영향")
    if num_rules_partially_skipped_body > 0:
        print(f"  - 바디 관계 ID 매핑 불가 등 이유로 건너뛴 개별 규칙 수: {num_rules_partially_skipped_body}개")


    print(f"\n--- 변환된 규칙 파일 저장 시작: {converted_rules_output_path} ---")
    # 6. 변환된 규칙 저장
    os.makedirs(os.path.dirname(converted_rules_output_path), exist_ok=True) # 출력 디렉토리 생성
    with open(converted_rules_output_path, 'w', encoding='utf-8') as f:
        json.dump(converted_rules_data, f, indent=4)
    print(f"변환된 규칙이 성공적으로 저장되었습니다: {converted_rules_output_path}")

if __name__ == '__main__':
    # --- 사용자 설정 필요 ---
    DATASET_NAME = "icews14"  # 또는 "icews18", "GDELT"

    # 1. SPARK 프로젝트의 어휘 파일 경로
    SPARK_VOCAB_DIR = f"./data/original/{DATASET_NAME}/" # SPARK 프로젝트 내 ./data/original/{DATASET_NAME}/ 경로
    
    # 2. LLM-DA 프로젝트가 규칙 생성 시 사용한 어휘 파일 경로
    #    LLM-DA 프로젝트의 datasets/{DATASET_NAME}/ 경로일 가능성이 높음
    LLMDA_VOCAB_DIR = f"./data/LLM-DA/{DATASET_NAME}/" # <<<< 실제 LLM-DA의 어휘 파일 경로로 수정하세요!

    # 3. LLM-DA가 생성한 원본 규칙 파일 경로 (예: confidence.json)
    LLMDA_RULES_INPUT_FILE = f"./data/LLM-DA/ranked_rules/{DATASET_NAME}/confidence.json" # <<<< 실제 LLM-DA 규칙 파일 경로로 수정하세요!
    
    # 4. 변환된 규칙을 저장할 출력 파일 경로 (SPARK 프로젝트 내에 저장 권장)
    CONVERTED_RULES_OUTPUT_FILE = f"./data/converted_rules_for_spark/{DATASET_NAME}/llmda_rules_spark_ids.json" # SPARK 프로젝트 내 저장 경로
    # --- 사용자 설정 끝 ---

    print(f"SPARK 어휘 경로: {SPARK_VOCAB_DIR}")
    print(f"LLM-DA 어휘 경로: {LLMDA_VOCAB_DIR}")
    print(f"LLM-DA 원본 규칙 파일: {LLMDA_RULES_INPUT_FILE}")
    print(f"변환된 규칙 저장 경로: {CONVERTED_RULES_OUTPUT_FILE}")

    convert_rules(
        spark_ent2id_path=os.path.join(SPARK_VOCAB_DIR, "entity2id.json"),
        spark_rel2id_path=os.path.join(SPARK_VOCAB_DIR, "relation2id.json"),
        llmda_ent2id_path=os.path.join(LLMDA_VOCAB_DIR, "entity2id.json"),
        llmda_rel2id_path=os.path.join(LLMDA_VOCAB_DIR, "relation2id.json"),
        llmda_rules_input_path=LLMDA_RULES_INPUT_FILE,
        converted_rules_output_path=CONVERTED_RULES_OUTPUT_FILE
    )
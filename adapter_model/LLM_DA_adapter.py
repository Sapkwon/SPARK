# llm_da_adapter.py

import torch
import torch.nn as nn
import os
import json
import numpy as np
import pandas as pd # LLM-DA의 rule_application.py가 pandas를 사용합니다.

# LLM-DA 프로젝트에서 필요한 함수/클래스들을 import 합니다.
# 이 파일들은 SPARK 프로젝트 내에서 접근 가능한 위치에 있어야 합니다.
# (예: SPARK 프로젝트 루트에 LLM_DA_code 폴더를 만들고 그 안에 LLM-DA 파일들을 넣어둔 후,
#  from LLM_DA_code.rule_application import ... 와 같이 경로 지정)
# 여기서는 같은 디렉토리 또는 PYTHONPATH에 있다고 가정합니다.
# 실제 파일 위치에 따라 이 import 경로는 반드시 수정해야 합니다!

# === LLM-DA 모듈 임포트 시작 ===
# 주의: 아래 파일들이 SPARK 프로젝트 내에 올바르게 위치하고, 경로가 맞아야 합니다.
# LLM-DA의 grapher.py, temporal_walk.py 등은 SPARK의 것과 역할이 겹칠 수 있으므로,
# 규칙 적용에 필요한 최소한의 함수만 가져오거나, 이름 충돌을 피하도록 주의합니다.

try:
    # LLM-DA의 규칙 적용 및 점수 계산 로직 (가장 중요)
    from rule_application import match_body_relations, get_walks, get_candidates
    from score_functions import score_12 # LLM-DA에서 사용하는 주된 점수 함수로 가정 (또는 다른 함수)
    from reasoning import calculate_scores as calculate_llm_da_candidate_scores # 이름 변경하여 충돌 방지
    from reasoning import load_rules as load_llm_da_rules_from_file # 이름 변경
    
    # LLM-DA의 TKG 데이터(엣지)를 준비하는 데 필요할 수 있음
    from temporal_walk import store_edges as store_llm_da_edges # 이름 변경
except ImportError as e:
    print(f"LLM-DA 모듈 임포트 중 오류 발생: {e}")
    print("LLM-DA의 rule_application.py, score_functions.py, reasoning.py, temporal_walk.py 등의 파일이")
    print("Python이 찾을 수 있는 경로에 있고, 필요한 함수들이 해당 파일에 정의되어 있는지 확인해주세요.")
    print("SPARK 프로젝트 내에 LLM-DA 코드 파일을 특정 폴더(예: 'llm_da_src')에 넣고,")
    print("from llm_da_src.rule_application import ... 와 같이 경로를 지정하는 것을 권장합니다.")
    raise # 오류 발생시켜서 확인하고 수정하도록 함
# === LLM-DA 모듈 임포트 끝 ===


class LLMDA_Adapter(nn.Module):
    def __init__(self, args_spark, vocab_dict_spark, tkg_facts_for_adapter, llm_da_ranked_rules_file_path):
        super().__init__()
        self.args_spark = args_spark # SPARK의 args
        self.vocab_spark = vocab_dict_spark # SPARK의 ent2id, id2rel 등
        
        print(f"LLMDA_Adapter: Initializing...")
        print(f"  Loading LLM-DA rules from: {llm_da_ranked_rules_file_path}")
        
        # 1. LLM-DA 규칙 로드
        # LLM-DA의 reasoning.py에 있는 load_rules 함수 활용 (여기서는 load_llm_da_rules_from_file로 임포트)
        # load_llm_da_rules_from_file(rules_file, dir_path) 형태이므로 dir_path도 전달
        self.llm_da_rules_dict = load_llm_da_rules_from_file(
            os.path.basename(llm_da_ranked_rules_file_path), # rules_file (파일명만)
            os.path.dirname(llm_da_ranked_rules_file_path)   # dir_path (파일이 있는 디렉토리)
        )
        if not self.llm_da_rules_dict:
            raise ValueError(f"Failed to load rules from {llm_da_ranked_rules_file_path} or rules are empty.")
        num_loaded_rules = sum(len(v) for v in self.llm_da_rules_dict.values())
        print(f"  Successfully loaded {num_loaded_rules} LLM-DA rules for {len(self.llm_da_rules_dict)} relations.")

        # 2. LLM-DA 규칙 적용에 필요한 TKG 엣지 정보 준비
        # tkg_facts_for_adapter는 SPARK로부터 받은 (h,r,t,ts) 형태의 Nx4 numpy 배열이어야 합니다.
        # 이 배열을 LLM-DA의 rule_application.py가 사용하는 edges 포맷으로 변환합니다.
        # LLM-DA의 temporal_walk.store_edges(quads) 함수가 {rel_id: np.array_of_edges_for_that_rel} 형식으로 만듭니다.
        if tkg_facts_for_adapter is not None:
            print(f"  Preparing TKG edges for LLM-DA rule application from {tkg_facts_for_adapter.shape[0]} facts...")
            self.tkg_edges_for_llm_da = store_llm_da_edges(tkg_facts_for_adapter)
            print(f"  Prepared edges for {len(self.tkg_edges_for_llm_da)} relations.")
        else:
            raise ValueError("LLMDA_Adapter requires TKG facts (tkg_facts_for_adapter) for rule application.")

        # 3. LLM-DA reasoning.py에서 사용하는 score_func 및 관련 파라미터 설정
        #    SPARK의 args 객체에 LLM-DA용 파라미터를 추가하고 여기서 참조합니다. (예: args_spark.LLMDA_lmbda)
        self.llm_da_score_func = score_12 # LLM-DA의 score_functions.py에 있는 함수
        self.llm_da_score_args_list = [ # reasoning.py의 main()에서 args 변수를 만드는 부분을 참고하여 구성
            self.args_spark.LLMDA_lmbda if hasattr(self.args_spark, 'LLMDA_lmbda') else 0.1,
            self.args_spark.LLMDA_weight0 if hasattr(self.args_spark, 'LLMDA_weight0') else 0.5, # score_12의 'a' 파라미터
            self.args_spark.LLMDA_confidence_type if hasattr(self.args_spark, 'LLMDA_confidence_type') else 'Common',
            self.args_spark.LLMDA_weight if hasattr(self.args_spark, 'LLMDA_weight') else 0.0, # score1 내부의 weight
            self.args_spark.LLMDA_min_conf if hasattr(self.args_spark, 'LLMDA_min_conf') else 0.01, # score1 내부 (여기선 안쓰임)
            self.args_spark.LLMDA_coor_weight if hasattr(self.args_spark, 'LLMDA_coor_weight') else 0.0
        ]
        # reasoning.py의 apply_rules는 args를 여러 개 받을 수 있도록 리스트로 처리하므로, 여기서도 리스트의 리스트로 만듭니다.
        self.llm_da_score_args_sets = [self.llm_da_score_args_list]


        # LLM-DA의 reasoning.py > parse_arguments() 에서 참조할 만한 다른 파라미터들
        # self.llm_da_reasoning_params = {
        #     "score_type": self.args_spark.LLMDA_score_type if hasattr(self.args_spark, 'LLMDA_score_type') else "noisy-or",
        #     "is_relax_time": self.args_spark.LLMDA_is_relax_time if hasattr(self.args_spark, 'LLMDA_is_relax_time') else False,
        #     "is_return_timestamp": False, # SPARK 어댑터는 최종 점수 분포만 필요
        #     "evaluation_type": "origin" # LLM-DA reasoning.py 참고
        # }
        # 위 파라미터들은 get_candidates 호출 시 직접 전달하거나, args 객체에 임시로 설정 후 전달할 수 있습니다.
        # 여기서는 간단하게 필요한 값들만 사용하겠습니다.

        print("LLMDA_Adapter initialized successfully.")

    def forward(self, batch_data):
        batch_queries_spark = batch_data["batch_queries"] # SPARK 쿼리: (ts, head_id, rel_id)
        bsz = batch_queries_spark.size(0)
        num_entities = len(self.vocab_spark["ent2id"])
        
        batch_adapter_distribution = torch.zeros(bsz, num_entities).to(self.args_spark.DEVICE)

        for i in range(bsz):
            query_ts_spark = batch_queries_spark[i, 0].item()
            query_head_id_spark = batch_queries_spark[i, 1].item()
            query_rel_id_spark = batch_queries_spark[i, 2].item()
            
            # LLM-DA의 reasoning.py는 쿼리를 (head, rel, dummy_tail, ts) 형태의 numpy array로 사용합니다.
            # 또한, relation ID를 문자열이 아닌 정수 ID로 사용하는지 확인 필요.
            # 여기서는 SPARK의 ID를 그대로 사용한다고 가정합니다.
            current_query_for_llm_da = np.array([query_head_id_spark, query_rel_id_spark, -1, query_ts_spark])

            candidate_scores_for_query_i = {} # 이 쿼리에 대한 {ent_id: 최종_점수}

            # LLM-DA의 reasoning.py > apply_rules 함수 핵심 로직 구현 시작
            if query_rel_id_spark in self.llm_da_rules_dict:
                rules_for_this_relation = self.llm_da_rules_dict[query_rel_id_spark]
                
                # 현재 쿼리 시간에 맞는 TKG 엣지 가져오기 (LLM-DA의 get_window_edges 와 유사)
                # 여기서는 __init__에서 준비한 self.tkg_edges_for_llm_da를 사용한다고 가정.
                # 실제로는 query_ts_spark를 기준으로 필터링된 엣지를 사용해야 할 수 있음 (LLM-DA reasoning.py의 windown_subgraph)
                # 지금은 self.tkg_edges_for_llm_da 가 이미 적절히 필터링/준비되었다고 가정합니다.
                current_window_edges = self.tkg_edges_for_llm_da

                # reasoning.py apply_rules의 cands_dict, timestamp_dict 와 유사한 구조
                cands_dict_list_for_query = [dict() for _ in self.llm_da_score_args_sets]
                timestamp_dict_list_for_query = [dict() for _ in self.llm_da_score_args_sets] # 사용 안 할 예정

                for rule_info_llm_da in rules_for_this_relation:
                    # rule_info_llm_da 형식: {"head_rel": ..., "body_rels": ..., "var_constraints": ..., "conf": ...}
                    
                    walk_edges = match_body_relations(
                        rule_info_llm_da, 
                        current_window_edges, 
                        current_query_for_llm_da, 
                        is_sample=(self.args_spark.LLMDA_is_sample if hasattr(self.args_spark, 'LLMDA_is_sample') else False)
                    )

                    if 0 not in [len(x) for x in walk_edges]:
                        rule_walks_df = get_walks(
                            rule_info_llm_da, 
                            walk_edges, 
                            is_relax_time=(self.args_spark.LLMDA_is_relax_time if hasattr(self.args_spark, 'LLMDA_is_relax_time') else False)
                        )

                        if rule_info_llm_da.get("var_constraints"): # var_constraints 키 존재 확인
                             # LLM-DA의 rule_application.py 에는 check_var_constraints 함수가 있지만, 여기서는 get_candidates가 처리한다고 가정
                             # rule_walks_df = check_var_constraints(rule_info_llm_da["var_constraints"], rule_walks_df)
                             pass


                        if not rule_walks_df.empty:
                            get_candidates( # rule_application.py의 함수
                                rule_info_llm_da,
                                rule_walks_df,
                                query_ts_spark,
                                cands_dict_list_for_query, # 이 딕셔너리가 내부적으로 업데이트됨
                                self.llm_da_score_func,
                                self.llm_da_score_args_sets,
                                list(range(len(self.llm_da_score_args_sets))), # dicts_idx
                                0,   # corre
                                False, # is_return_timestamp
                                'origin', # evaluation_type (LLM-DA reasoning.py 참고)
                                timestamp_dict_list_for_query # 사용 안 함
                            )
                
                # 여러 규칙으로부터 집계된 후보 점수 (첫 번째 score_args 세트 사용 가정)
                if cands_dict_list_for_query[0]:
                    # reasoning.py의 calculate_scores 함수 (여기서는 calculate_llm_da_candidate_scores로 임포트)
                    final_scores_for_candidates = calculate_llm_da_candidate_scores(
                        cands_dict_list_for_query[0], 
                        {"score_type": self.args_spark.LLMDA_score_type if hasattr(self.args_spark, 'LLMDA_score_type') else "noisy-or"}
                    )
                    candidate_scores_for_query_i = dict(zip(cands_dict_list_for_query[0].keys(), final_scores_for_candidates))
            # LLM-DA 규칙 기반 추론 로직 끝

            # 후보 점수를 현재 배치의 해당 샘플에 대한 전체 엔티티 분포로 변환
            if candidate_scores_for_query_i:
                for ent_id, score_val in candidate_scores_for_query_i.items():
                    if 0 <= ent_id < num_entities: # 유효한 엔티티 ID이고, SPARK의 ID 체계와 맞는지 확인
                        batch_adapter_distribution[i, ent_id] = float(score_val) 
                
                # (선택적) 분포 정규화: LLM-DA의 점수 범위에 따라 필요할 수 있음
                # 예: 모든 엔티티에 대해 점수가 있으므로 소프트맥스 적용
                # if torch.sum(batch_adapter_distribution[i]) > 0 : # 0으로만 채워진 경우 제외
                #    batch_adapter_distribution[i] = torch.softmax(batch_adapter_distribution[i], dim=-1)
                # else: # 모든 점수가 0이면 (후보가 없으면) 균등 분포 등으로 처리
                #    batch_adapter_distribution[i, :] = 1.0 / num_entities

                # 또는, SPARK의 MainModel에서 결합 시 정규화가 없다면,
                # 여기서 나온 점수들이 LLM의 logit/확률과 유사한 스케일을 갖도록 조정 필요 가능성 있음
                # 지금은 원시 점수를 그대로 전달한다고 가정
                pass

        return batch_adapter_distribution.to(self.args_spark.DEVICE)
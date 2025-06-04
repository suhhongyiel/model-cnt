# -*- coding: utf-8 -*-
"""
activity_sleep_llm_pipeline.py (v3)
──────────────────────────────────
CSV(활동+수면) → CatBoost 회귀 + SHAP → Llama-3 8B GGUF 한국어 코칭

🍃 2025-06-04 UPDATE: LLM 입력을 **최근 N일(기본 7일)** 로 축소해
토큰 초과 오류를 방지합니다. CatBoost 학습과 SHAP 계산은 전체 데이터로 유지됩니다.

설치
-----
    pip install pandas catboost shap llama-cpp-python

CLI 예시
--------
    python activity_sleep_llm_pipeline.py data.csv --model llama.gguf --window 7
"""

import argparse
from pathlib import Path
import textwrap
from typing import List, Dict

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import shap
from llama_cpp import Llama

# ──────────────────────────────────────────────────────────────
# 1. CSV 로드 & 전처리
# ──────────────────────────────────────────────────────────────

def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df

# ──────────────────────────────────────────────────────────────
# 2. CatBoost + SHAP
# ──────────────────────────────────────────────────────────────

def get_feature_matrix(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """타깃 제외 + 숫자형 열만 반환"""
    return df.drop(columns=[target]).select_dtypes(include="number")


def train_model(df: pd.DataFrame, target: str = "efficiency") -> CatBoostRegressor:
    X = get_feature_matrix(df, target)
    y = df[target]
    model = CatBoostRegressor(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        loss_function="RMSE",
        silent=True,
    )
    model.fit(X, y)
    return model


def make_shap_summary(model: CatBoostRegressor, X: pd.DataFrame, top_k: int = 5) -> List[Dict]:
    """전체 X로 SHAP 계산 후 마지막 샘플 기준 상위 피처 반환"""
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)
    last_vals = shap_vals[-1]
    idx = np.abs(last_vals).argsort()[::-1][:top_k]
    return [
        {
            "feature": X.columns[i],
            "impact": float(last_vals[i]),
            "value": float(X.iloc[-1, i]),
        }
        for i in idx
    ]

# ──────────────────────────────────────────────────────────────
# 3. LLM 프롬프트 & 호출
# ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent(
    """
    당신은 전문 수면·활동 코치이자 생체신호 데이터 분석가입니다.

    🔴 **절대 규칙** 🔴
    1. 오직 한국어만 사용하십시오. 영어·숫자 혼합 표기, 로마자, 이모티콘 모두 금지.
    2. 출력 형식 의 양식을 따라서 진행 해주세요.

    [출력 형식]
    [advice] 수면의 질을 높이기 위한 핵심 실천 조언 구제적으로 제시 2~3문장
    [why] 서로 다른 지표를 최소 3개 이상, 실제 숫자를 포함해 개선 사항을 위한 수치적 근거 설명 2문장
    [ask] 추가로 확인하고 싶은 질문 1문장
    """
)

USER_PROMPT_TEMPLATE = textwrap.dedent(
    """
    ### 최근 {n_days}일 원본 데이터
    ```
    {table}
    ```

    ### 분석 요약 (전체 데이터 기반)
    - CatBoost 예측 수면 효율: {pred:.1f}%
    - SHAP 영향 상위 {k}개 피처:
    {shap_lines}

    위 정보를 바탕으로 조언을 작성해 주세요.
    """
)


def build_user_prompt(df_last: pd.DataFrame, pred: float, shap_top: List[Dict]) -> str:
    table = df_last.to_string(index=False)
    shap_lines = "\n".join(
        f"  * {d['feature']}: 값 {d['value']:.2f}, 영향 {d['impact']:+.2f}" for d in shap_top
    )
    return USER_PROMPT_TEMPLATE.format(
        n_days=len(df_last),
        table=table,
        pred=pred,
        k=len(shap_top),
        shap_lines=shap_lines,
    )


def run_llm(model_path: Path, system_prompt: str, user_prompt: str, n_threads: int = 8) -> str:
    llm = Llama(
        model_path=str(model_path),
        n_threads=n_threads,
        n_ctx=4096,
        temperature=0.5,
    )
    result = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stop=["[ask]"]
    )
    return result["choices"][0]["message"]["content"].strip()

# ──────────────────────────────────────────────────────────────
# 4. 메인
# ──────────────────────────────────────────────────────────────

def main(csv: Path, model_path: Path, window: int):
    df = load_data(csv)

    # ── 2-1) CatBoost 학습 & 예측 (전체 데이터 사용)
    model = train_model(df, target="efficiency")
    X_full = get_feature_matrix(df, "efficiency")
    pred_eff = float(model.predict(X_full.iloc[[-1]])[0])
    shap_top = make_shap_summary(model, X_full)

    # ── 2-2) LLM 에는 최근 window 일만 전달
    df_last = df.tail(window)

    user_prompt = build_user_prompt(df_last, pred_eff, shap_top)
    advice = run_llm(model_path, SYSTEM_PROMPT, user_prompt)

    print("\n=== LLM 한국어 코칭 메시지 ===\n")
    print(advice)

# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="활동+수면 CSV → CatBoost+LLM 코칭")
    parser.add_argument("--csv", type=Path, help="CSV 파일 경로")
    parser.add_argument("--model", type=Path, required=True, help="GGUF 모델 경로")
    parser.add_argument("--window", type=int, default=7, help="LLM 입력에 사용할 최근 일수 (기본 7)")
    args = parser.parse_args()

    main(args.csv, args.model, args.window)

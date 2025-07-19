import pandas as pd

# 1) 원본 데이터 읽기
df = pd.read_csv("./data/train.csv")

# 2) generated == 0 필터링(filter)
human_df = df[df["generated"] == 0].copy()

# 3) full_text를 \n 기준으로 리스트(list)로 분할(split)
human_df["paragraphs"] = human_df["full_text"].str.split("\n")

# 4) 빈 문자열 제거 + strip(앞뒤 공백 제거)
human_df["paragraphs"] = human_df["paragraphs"].apply(
    lambda lst: [p.strip() for p in lst if p.strip()]
)

# 5) explode → 한 문단(문장)씩 행으로 펼치기
exploded = human_df.explode("paragraphs").reset_index(drop=True)

# 6) 문단 번호(paragraph_index) 매기기 – 같은 title 기준으로 누적
exploded["paragraph_index"] = exploded.groupby("title").cumcount()

exploded["generated"] = 0

# 7) 원하는 컬럼만 선택(select)하고 저장(save)
out_cols = ["title", "paragraph_index", "paragraphs", "generated"]
exploded[out_cols].to_csv("./data/train_paragraphs.csv", index=False)

print("완료! ➜ train_human_paragraphs.csv")

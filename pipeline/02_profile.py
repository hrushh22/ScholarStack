"""Step 2 - Profile & validate: stage postings to Parquet, then profile every source.

Output: data/interim/postings_raw/*.parquet, reports/results/data_profile.json
"""
from scholarstack.clean import stage_postings
from scholarstack.profiling import profile_all

if __name__ == "__main__":
    stage_postings()
    rep = profile_all()
    print(f"resumes:  {rep['resumes']['rows']:,} rows, {rep['resumes']['Resume_str_duplicate_texts']} duplicate texts")
    print(f"postings: {rep['postings']['rows']:,} rows, salary null rate "
          f"{rep['postings']['null_rate']['normalized_salary']:.0%}")
    print(f"O*NET:    {rep['onet']['Occupation Data']['rows']:,} occupations")

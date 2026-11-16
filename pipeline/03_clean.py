"""Step 3 - Clean & filter: resumes (HTML -> sections), postings (Dask), O*NET profiles.

Output: data/processed/{resumes.parquet, postings/, onet_occupations.parquet}
"""
from dask.distributed import Client, LocalCluster

from scholarstack.clean import build_onet_profiles, clean_postings, clean_resumes

if __name__ == "__main__":
    clean_resumes()
    build_onet_profiles()
    with LocalCluster(n_workers=4, threads_per_worker=1, memory_limit="3GB") as cluster, Client(cluster) as client:
        print("Dask dashboard:", client.dashboard_link)
        clean_postings()

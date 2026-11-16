-- ScholarStack serving database (Supabase Postgres + pgvector).
-- Loaded by pipeline/11_load_database.py. Safe to re-run.

create extension if not exists vector with schema extensions;

-- ---------------------------------------------------------------------------
-- Reference corpus: resume chunks with their sentence-transformer embeddings.
-- ---------------------------------------------------------------------------
create table if not exists public.resume_chunks (
    chunk_id      text primary key,
    resume_id     text not null,
    category      text not null,
    resume_title  text,
    section       text not null,
    content       text not null,
    skills        text[] not null default '{}',
    embedding     extensions.vector(384) not null
);
create index if not exists resume_chunks_category_idx on public.resume_chunks (category);
create index if not exists resume_chunks_embedding_idx
    on public.resume_chunks using hnsw (embedding extensions.vector_ip_ops);

-- ---------------------------------------------------------------------------
-- O*NET occupation profiles + LinkedIn demand aggregated per occupation.
-- ---------------------------------------------------------------------------
create table if not exists public.onet_occupations (
    code                text primary key,
    title               text not null,
    description         text,
    job_zone            int,
    essential_skills    jsonb not null default '[]',
    transferable_skills jsonb not null default '[]',
    knowledge           jsonb not null default '[]',
    tasks               jsonb not null default '[]',
    technologies        jsonb not null default '[]',
    market              jsonb,
    embedding           extensions.vector(384) not null
);

-- ---------------------------------------------------------------------------
-- Every live generation is logged (no applicant resume text is stored).
-- ---------------------------------------------------------------------------
create table if not exists public.runs (
    id             uuid primary key default gen_random_uuid(),
    created_at     timestamptz not null default now(),
    mode           text not null,
    job_title      text,
    jd_chars       int,
    has_resume     boolean not null default false,
    category       text,
    k              int,
    model          text,
    retrieved_ids  text[],
    onet_code      text,
    output         jsonb,
    latency_ms     int,
    prompt_tokens  int,
    output_tokens  int
);

-- Human ratings collected in the web app (the "human" half of the rating study).
create table if not exists public.feedback (
    id          bigint generated always as identity primary key,
    created_at  timestamptz not null default now(),
    run_id      uuid references public.runs (id) on delete cascade,
    system      text not null check (system in ('rag', 'llm_only', 'baseline')),
    rating      int not null check (rating between 1 and 5),
    comment     text check (char_length(comment) <= 1000)
);

-- ---------------------------------------------------------------------------
-- Security: the browser (anon key) can only call the functions below and add
-- feedback. Tables are not directly readable; the Edge Function uses the
-- service role.
-- ---------------------------------------------------------------------------
alter table public.resume_chunks    enable row level security;
alter table public.onet_occupations enable row level security;
alter table public.runs             enable row level security;
alter table public.feedback         enable row level security;

grant usage on schema public to anon, authenticated;
grant insert on public.feedback to anon, authenticated;
drop policy if exists "anyone can rate" on public.feedback;
create policy "anyone can rate" on public.feedback for insert to anon, authenticated with check (true);

-- Top-k resume chunks by inner product (= cosine; embeddings are normalised).
-- Unfiltered queries use the HNSW index; category-filtered queries do an exact
-- scan over that category (~1k rows), which avoids HNSW post-filtering losses.
create or replace function public.match_resume_chunks(
    query_embedding extensions.vector(384),
    match_count     int  default 8,
    filter_category text default null
)
returns table (chunk_id text, resume_id text, category text, resume_title text, section text,
               content text, skills text[], similarity float)
language plpgsql stable security definer
set search_path = public, extensions
as $$
begin
    match_count := least(greatest(match_count, 1), 30);
    if filter_category is null then
        return query
        select c.chunk_id, c.resume_id, c.category, c.resume_title, c.section, c.content, c.skills,
               (-(c.embedding <#> query_embedding))::float
        from public.resume_chunks c
        order by c.embedding <#> query_embedding
        limit match_count;
    else
        return query
        select c.chunk_id, c.resume_id, c.category, c.resume_title, c.section, c.content, c.skills,
               (-(c.embedding <#> query_embedding))::float
        from public.resume_chunks c
        where c.category = filter_category
        order by (c.embedding <#> query_embedding) + 0   -- "+ 0" = exact scan, not the HNSW index
        limit match_count;
    end if;
end;
$$;

create or replace function public.match_occupations(
    query_embedding extensions.vector(384),
    match_count     int default 3
)
returns table (code text, title text, description text, job_zone int, essential_skills jsonb,
               transferable_skills jsonb, knowledge jsonb, tasks jsonb, technologies jsonb,
               market jsonb, similarity float)
language sql stable security definer
set search_path = public, extensions
as $$
    select o.code, o.title, o.description, o.job_zone, o.essential_skills, o.transferable_skills,
           o.knowledge, o.tasks, o.technologies, o.market, (-(o.embedding <#> query_embedding))::float
    from public.onet_occupations o
    order by o.embedding <#> query_embedding
    limit least(greatest(match_count, 1), 10);
$$;

-- Aggregate, anonymous usage numbers for the dashboard.
create or replace function public.usage_stats()
returns json
language sql stable security definer
set search_path = public
as $$
    select json_build_object(
        'runs', (select count(*) from public.runs),
        'ratings', (select count(*) from public.feedback),
        'avg_rating', (select json_object_agg(system, avg_rating) from
                         (select system, round(avg(rating)::numeric, 2) as avg_rating
                          from public.feedback group by system) s)
    );
$$;

revoke all on function public.match_resume_chunks(extensions.vector, int, text) from public;
revoke all on function public.match_occupations(extensions.vector, int) from public;
grant execute on function public.match_resume_chunks(extensions.vector, int, text) to anon, authenticated, service_role;
grant execute on function public.match_occupations(extensions.vector, int) to anon, authenticated, service_role;
grant execute on function public.usage_stats() to anon, authenticated, service_role;

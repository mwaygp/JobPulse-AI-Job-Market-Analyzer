-- Run against data/jobpulse.db  (sqlite3 data/jobpulse.db < sql/queries.sql)

-- 1. Most requested skills overall
SELECT skill, COUNT(*) AS postings,
       ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM jobs), 1) AS pct
FROM job_skills GROUP BY skill ORDER BY postings DESC LIMIT 15;

-- 2. Most requested programming skills per role
SELECT j.title, s.skill, COUNT(*) AS postings
FROM jobs j JOIN job_skills s ON s.job_id = j.id
WHERE s.skill IN ('python','java','javascript','typescript','sql','go','c++','c#','rust','scala','r')
GROUP BY j.title, s.skill
ORDER BY j.title, postings DESC;

-- 3. Median-ish pay by skill (average, SQLite has no MEDIAN)
SELECT s.skill, ROUND(AVG(j.salary)) AS avg_salary, COUNT(*) AS postings
FROM jobs j JOIN job_skills s ON s.job_id = j.id
WHERE j.salary IS NOT NULL
GROUP BY s.skill HAVING postings >= 15
ORDER BY avg_salary DESC LIMIT 15;

-- 4. Skills that most often appear together with python
SELECT b.skill, COUNT(*) AS together
FROM job_skills a JOIN job_skills b ON a.job_id = b.job_id AND a.skill <> b.skill
WHERE a.skill = 'python'
GROUP BY b.skill ORDER BY together DESC LIMIT 10;

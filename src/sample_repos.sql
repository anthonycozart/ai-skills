-- Sample repos with push activity in March 2026
-- Source: GHArchive public dataset on Google BigQuery
-- Filters to repos with >= 3 push events in the month
-- Ordering by FARM_FINGERPRINT provides a deterministic pseudo-random order
-- so we can take reproducible slices with LIMIT

SELECT repo_id
FROM (
  SELECT repo.id AS repo_id
  FROM `githubarchive.month.202603`
  WHERE type = 'PushEvent'
)
GROUP BY repo_id
HAVING COUNT(*) >= 3
ORDER BY FARM_FINGERPRINT(CAST(repo_id AS STRING))
LIMIT 250000

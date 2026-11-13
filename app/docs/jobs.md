# Jobs
Everything long runs as a background job: lanes, progress, statuses, cancel and retry.

- Importing images, downloading and loading models, auto-annotating and workflow runs are jobs. They show in the jobs drawer with progress; /jobs/ID has a job's log.
- **Lanes:** io (imports, downloads) and gpu (auto-annotate, runs, loading models). A lane runs one job at a time, oldest first, so an import never waits for a run.
- **Statuses:** queued → running → succeeded, failed, cancelled or interrupted (the app closed or crashed while it was queued or running).
- **Cancel** stops a job between items; what it did so far stays. **Retry** starts a new job with the same settings; it skips what's done (an import skips images it has).
- Nothing restarts by itself.
- You follow a job with get_job or wait_for_job.

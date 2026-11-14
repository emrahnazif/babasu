# Jobs
Everything long runs as a background job: lanes, progress, statuses, cancel and retry.

- Importing images, downloading and loading models, auto-annotating and workflow runs are jobs. They show in the jobs drawer with progress; /jobs/ID has a job's log.
- **Lanes:** io (imports, downloads) and gpu (auto-annotate, runs, loading models). A lane runs one job at a time, oldest first, so an import never waits for a run.
- **Waiting:** a job can wait for another, e.g. a run for the download of a model it needs. It stays queued ("Waiting for job N") while other jobs run; if the one it waits for fails or is cancelled, it ends the same way and says so.
- **Statuses:** queued → running → succeeded, failed, cancelled or interrupted (the app closed or crashed while it was queued or running).
- **Cancel** stops a job between items; what it did so far stays. **Retry** starts a new job with the same settings; it skips what's done (an import skips images it has).
- Nothing restarts by itself.
- You follow a job with wait_for_job, or GET /api/jobs/ID.

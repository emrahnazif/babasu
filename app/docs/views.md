# Views
Pages written by an agent: reports that keep what they showed, and Studio pages; versions and the navigation.

- A view is a Jinja page in the main area, built from the app's components and its API (api()). get_components has how to write one.
- **Report** (the default): it keeps what it showed. When a version is saved, every API answer it read is recorded with it, and the page replays them: a report doesn't change when the data does. To refresh it, save it again.
- **Studio**: a live page for one workflow; it reads fresh data each time and replaces the workflow's plain Studio page.
- Every save is a new **version**; older ones can be seen and made the newest again. Deleting a view removes all its versions.
- The **navigation** (sidebar) can have links added after the app's own sections: to a view, or any page of the app.
- Views stay separate from the app's own pages: they can't change them.

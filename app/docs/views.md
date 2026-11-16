# Views
Pages written by an agent; versions and the navigation.

- A view is a Jinja page in the main area, built from the app's components and its API (api()). get_components has how to write one.
- A view can also be given data: what an agent's code worked out (counts, scores, chosen images), saved with the view from that code (read_docs code), so its numbers come from the code, not typed in.
- A view keeps what it showed. When a version is saved, every API answer it read is recorded with it, and its data kept; the page replays them: a view doesn't change when the data does. To refresh it, save it again.
- Every save is a new **version**; older ones can be seen and made the newest again. Deleting a view removes all its versions.
- The **navigation** (sidebar) can have links added after the app's own sections: to a view, or any page of the app.
- Views stay separate from the app's own pages: they can't change them.

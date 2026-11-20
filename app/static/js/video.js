// A video on its image page, a worked frame at a time: the frame is a picture of its own
// (/api/images/{id}/frames/{n}), so the annotation editor and the click tool work on it as on an
// image. The page renders the frame asked for and the timeline; this steps through the frames:
//   <div data-video data-frames="[{frame, time (seconds)}, …]" data-frame="12" data-duration="9.5"
//        data-picture="/api/images/7/frames/">
//     [data-step="-1"] [data-step="1"]   previous and next frame (← →, shortcuts.js)
//     [data-play]                        play and pause, at the rate the video is worked at
//     [data-clock]                       the frame's moment
//     [data-tracks]                      the timeline: a click goes to the frame at that moment
//       [data-playhead]                  where the frame shown is (--at, 0..1)
//     [name=sample_fps]                  the rate it's worked at: its form posts on change
//   </div>
// A frame is shown once its picture is in (then "frame", detail {frame, next}, on document, for the
// editor to follow), and the next one is fetched ahead. The address keeps the frame (?frame=), so
// a reload, or a part of the page refreshed (js/refresh-region.js), shows the same one.
const video = document.querySelector("[data-video]");

if (video) {
  const frames = JSON.parse(video.dataset.frames);
  const duration = Number(video.dataset.duration);
  const picture = document.querySelector("[data-markin-stage] img");
  const play = video.querySelector("[data-play]");
  let at = Math.max(0, frames.findIndex((f) => f.frame === Number(video.dataset.frame)));
  let playing = null; // the timer of the next frame while it plays
  let wanted = at; // the frame last asked for: an older one whose picture comes in later isn't shown

  const url = (index) => video.dataset.picture + frames[index].frame;
  // As formatting.clock(seconds, tenths=True) writes it: "0:42.5", "1:02:09.0".
  const clock = (seconds) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const rest = (seconds % 60).toFixed(1).padStart(4, "0");
    return hours ? `${hours}:${String(minutes).padStart(2, "0")}:${rest}` : `${minutes}:${rest}`;
  };

  // The frame at `index`, once its picture is in; resolves when it's shown (false if overtaken).
  function show(index) {
    wanted = index;
    const next = new Image();
    next.src = url(index);
    return next.decode().then(
      () => {
        if (wanted !== index) return false;
        at = index;
        picture.src = next.src;
        const { frame, time } = frames[index];
        video.querySelector("[data-playhead]").style.setProperty("--at", time / duration);
        video.querySelector("[data-clock]").textContent = clock(time);
        const address = new URL(location.href);
        address.searchParams.set("frame", frame);
        history.replaceState(null, "", address);
        announce();
        if (index + 1 < frames.length) new Image().src = url(index + 1); // fetched ahead
        return true;
      },
      () => false, // no picture (the app didn't answer): the frame shown stays
    );
  }

  // The frame shown, for the editor to follow (js/annotation-editor.js).
  const announce = () => {
    const detail = { frame: frames[at].frame, next: frames[at + 1]?.frame ?? null, playing: playing !== null };
    document.dispatchEvent(new CustomEvent("frame", { detail }));
  };

  const step = (by) => show(Math.min(frames.length - 1, Math.max(0, wanted + by)));

  function pause() {
    if (playing === null) return;
    clearTimeout(playing);
    playing = null;
    play.setAttribute("aria-pressed", "false");
    announce(); // stopped on it: the click tool gets it ready
  }

  // Each frame for as long as its stretch lasts, or longer while its picture is on its way.
  async function advance() {
    if (at + 1 >= frames.length) return pause();
    const started = performance.now();
    if (!(await show(at + 1)) || !playing) return;
    const lasts = ((frames[at + 1]?.time ?? duration) - frames[at].time) * 1000;
    playing = setTimeout(advance, Math.max(0, lasts - (performance.now() - started)));
  }

  play.addEventListener("click", () => {
    if (playing) return pause();
    play.setAttribute("aria-pressed", "true");
    if (at + 1 >= frames.length) at = -1; // from the start again
    playing = setTimeout(advance, 0);
  });
  for (const button of video.querySelectorAll("[data-step]")) {
    button.addEventListener("click", () => (pause(), step(Number(button.dataset.step))));
  }
  // The timeline is replaced when it's refreshed, so its clicks are listened for on the video.
  video.addEventListener("click", (event) => {
    const tracks = event.target.closest("[data-tracks]");
    if (!tracks) return;
    const lane = tracks.querySelector("[data-lane]").getBoundingClientRect();
    const time = ((event.clientX - lane.left) / lane.width) * duration;
    if (time < 0) return; // on a class's name
    pause();
    show(Math.max(0, frames.findLastIndex((f) => f.time <= time)));
  });
  // The rate it's worked at (partials/select.html's hidden input): posted with the frame shown.
  video.addEventListener("change", (event) => {
    if (event.target.name !== "sample_fps") return;
    const form = event.target.form;
    form.elements.frame.value = frames[at].frame;
    form.requestSubmit();
  });
}

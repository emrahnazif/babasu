# Filters
How a project's images are picked by what's in them: the filter's text form, used by the page, the API and your tools.

A filter is a list of terms, all of which must hold. The project page shows it as chips the person can change; its text form is in the page's address (?where=) and in your tools' `where`.

- `labelled`, `unlabelled`: has objects or not.
- `name:TEXT`: the file name contains TEXT.
- `objects:RANGE`: how many objects the image has.
- `(…)`: the image has an object with all of these at once; `-(…)`: it has none:
  - class names: `car`, or `car|truck` for either; names with spaces in quotes: `"traffic light"`.
  - `w:RANGE`, `h:RANGE`, `size:RANGE` (pixels; size is both), `area:RANGE`, `aspect:RANGE` (width / height).
  - `edge` (touches the image edge), `edge:no`.
- `car` is short for `(car)`; `-person` for `-(person)`.
- RANGE: `8..16`, `8..`, `..16`, `>8`, `<16`, `>=8`, `<=16` or `8`.

Examples:
- `labelled (car size:8..16) -person`: labelled images with a small car and no person.
- `(circle edge)`: images where a circle touches the edge.
- `objects:0`: images without objects.

A filter naming a class the project doesn't have is refused with why. Only active classes count (see projects: removing a class).

## Measuring
`describe` (and /api/projects/ID/stats) measures the images matching a filter: objects per class, the spread of object widths, heights, areas or aspect, or how many images have how many objects.

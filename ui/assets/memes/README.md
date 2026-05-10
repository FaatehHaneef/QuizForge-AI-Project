# Meme assets — folder convention

Drop any `.gif`, `.png`, `.jpg`, `.jpeg`, or `.webp` into the matching folder.
The Streamlit app picks one file at random per render — add as many as you want
per category for variety.

## Folder → screen mapping

| Folder | Screen / moment | Vibe |
|---|---|---|
| `welcome/` | Welcome / splash | side-eye / "why r u here" — `eh` |
| `article_input/` | Passage paste screen | suspicious — `neko sus` |
| `loading/` | After Submit, while inference runs | crashing out — `crash out` |
| `quiz_view/` | While answering the question | crying student — `crying cat` |
| `correct/` | Correct answer popup | yolo W cat — `yolo neko` |
| `wrong/` | Wrong answer popup | dazed defeat — `discombobulated cat` |
| `hints/` | Hint panel banner | argument — `woman vs cat` |
| `dashboard/` | Dev/analytics dashboard | crying wojak — `dostey` |
| `outro/` | "Submit project for grading" outro | beg for marks — `marks dedein please` |

## Filenames

Filenames within a folder are arbitrary — the app reads the directory at runtime.
For self-documentation, use the meme's nickname, e.g.:

```
welcome/eh.png
article_input/neko_sus.png
loading/crash_out.gif
quiz_view/crying_cat.jpg
correct/yolo_neko.gif
wrong/discombobulated_cat.jpg
hints/woman_vs_cat.jpg
dashboard/dostey.png
outro/marks_dedein_please.jpg
```

## Replacing or adding memes

- Drop in additional files → app rotates between them randomly.
- Delete a file → app picks from whatever's left.
- Empty folder → screen shows a dashed placeholder instructing you to drop a meme in.

No code changes required; the picker reads each folder fresh on every render.

import json

with open("fma_annotations.jsonl", encoding="utf-8") as f:
    for i, line in enumerate(f):
        item = json.loads(line)

        print("=" * 80)
        print("TRACK:", item["track_id"])
        print("GENRE:", item["analysis"]["genre"])
        print("MOOD:", item["analysis"]["mood"])
        print("THEMES:", item["analysis"]["themes"])
        print("PROMPT:", item["analysis"]["lyric_generation_prompt"])

        if i == 19:
            break
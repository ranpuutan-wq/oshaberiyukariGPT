from llm_openai import generate_turns

turns = generate_turns("夏の暑さについて。思うこと。希望", 8, model="gpt-4o-mini", temperature=1.1)
for t in turns:
    print(f"[{t['speaker']}] {t['text']}")

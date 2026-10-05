#!/usr/bin/env python3
"""context-me extractor.

Reads the N most recent Claude Code session transcripts and keeps only the
USER-side signal: every human prompt (harness-injected turns, system blocks,
and image bytes removed), slash commands, URLs, image counts, pasted-block
sizes, plus the tools / MCP servers / skills / CLI commands used per session.
Assistant prose and tool results are dropped.

Output is written as JSONL shards sized so that each shard fits in ONE Read
call of a reader agent (Claude Code's Read tool refuses files over 256 KB and
truncates lines over 2000 characters). Every shard ends with an "end" line so
a reader can prove it saw the whole file.
"""
import argparse
import glob
import json
import os
import re
from collections import Counter
from datetime import datetime

INJECTED = re.compile(
    r"<system-reminder>.*?</system-reminder>"
    r"|<local-command-caveat>.*?</local-command-caveat>"
    r"|<local-command-stdout>.*?</local-command-stdout>"
    r"|<command-message>.*?</command-message>"
    r"|<command-(?:name|args)>.*?</command-(?:name|args)>",
    re.S,
)
# Older transcripts lack origin/isMeta markers; these prefixes catch the same
# harness-generated "user" turns by their text.
HARNESS_PREFIX = re.compile(
    r"^\s*(?:<task-notification>"
    r"|This session is being continued from a previous conversation"
    r"|Another Claude session sent a message"
    r"|Base directory for this skill"
    r"|\[Request interrupted by user)"
)
CMD_NAME = re.compile(r"<command-name>(.*?)</command-name>", re.S)
CMD_ARGS = re.compile(r"<command-args>(.*?)</command-args>", re.S)
URL = re.compile(r"https?://[^\s\"'<>)]+")
DOMAIN = re.compile(r"https?://([^/\s\"'<>)]+)")
EXT = re.compile(
    r"\b[\w-]+\.(xlsx|xls|csv|pdf|docx|pptx|png|jpg|jpeg|json|yaml|yml|md|sql|ipynb|log|zip)\b",
    re.I,
)
PASTED_MIN = 1200   # chars; beyond this a prompt very likely contains pasted content
TEXT_CAP = 1500     # chars kept per prompt; keeps every JSONL line under Read's 2000-char limit
CLI_CAP = 200       # chars kept of a CLI command's first line
LEADING_CD = re.compile(r"^\s*cd\s+\S+\s*(?:&&|;)\s*")
LEADING_ENV = re.compile(r"^(?:\s*[A-Za-z_][A-Za-z0-9_]*=\S*\s*(?:;|&&)?\s*)+")
# Generic shell utilities say nothing about which product a user works with;
# commands like az, gh, curl, mongosh, docker, kubectl do.
SHELL_NOISE = {
    "cat", "sed", "grep", "rg", "ls", "echo", "head", "tail", "find", "cd", "rm", "cp", "mv",
    "mkdir", "wc", "awk", "sort", "uniq", "cut", "tr", "xargs", "for", "while", "if", "test",
    "true", "false", "sleep", "printf", "pwd", "touch", "chmod", "diff", "file", "stat", "du",
    "df", "which", "type", "export", "set", "unset", "source", ".", "tee", "less", "more", "[",
    "get-childitem", "get-content", "select-string", "write-output", "write-host", "set-location",
    "test-path", "remove-item", "copy-item", "new-item", "foreach", "try", "$",
}


def cli_signal(command):
    """First line of a CLI command with cd/env prefixes removed, or None if it is shell noise."""
    first = (command or "").strip().split("\n", 1)[0]
    first = LEADING_ENV.sub("", LEADING_CD.sub("", first)).strip()
    if not first:
        return None
    words = first.split()
    word = words[0].strip("\"'(&").lower()
    base = os.path.basename(word.replace("\\", "/"))
    # Interpreter runs (python script.py, .venv/.../python.exe) are Claude running
    # local code, not a product the user works with.
    if not base or base in SHELL_NOISE or word.startswith("$") or re.match(r"(python|py)(3|\.exe)?$", base):
        return None
    # The first few words identify the product and action (az webapp list,
    # gh pr view, git push); the rest is arguments that only fragment the counts.
    return " ".join([base] + words[1:4])[:CLI_CAP]


def is_harness_turn(o, raw_text):
    if o.get("isMeta") or o.get("isCompactSummary"):
        return True
    origin = o.get("origin")
    if isinstance(origin, dict) and origin.get("kind") not in (None, "human"):
        return True
    return bool(HARNESS_PREFIX.match(raw_text))


def scan(path):
    prompts, tools, mcp, skills, cli = [], Counter(), Counter(), Counter(), Counter()
    meta = {"cwd": None, "branch": None, "first_ts": None, "last_ts": None}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if o.get("isSidechain"):
                continue
            ts = o.get("timestamp")
            if ts:
                meta["first_ts"] = meta["first_ts"] or ts
                meta["last_ts"] = ts
            meta["cwd"] = meta["cwd"] or o.get("cwd")
            meta["branch"] = meta["branch"] or o.get("gitBranch")
            m = o.get("message")
            if not isinstance(m, dict):
                continue
            content = m.get("content")
            if o.get("type") == "user":
                texts, images, has_tool_result = [], 0, False
                if isinstance(content, str):
                    texts.append(content)
                elif isinstance(content, list):
                    for b in content:
                        bt = b.get("type")
                        if bt == "text":
                            texts.append(b.get("text", ""))
                        elif bt == "image":
                            images += 1
                        elif bt == "tool_result":
                            has_tool_result = True
                if has_tool_result and not texts:
                    continue  # pure tool result, not a human prompt
                raw = "\n".join(texts)
                if is_harness_turn(o, raw):
                    continue
                cmds = [c.strip() for c in CMD_NAME.findall(raw)]
                args = CMD_ARGS.findall(raw)
                clean = INJECTED.sub("", raw).strip()
                if not clean and not images and not cmds:
                    continue
                prompts.append(
                    {
                        "idx": len(prompts),
                        "ts": ts,
                        "chars": len(clean),
                        "truncated": len(clean) > TEXT_CAP,
                        "pasted_block": len(clean) >= PASTED_MIN,
                        "images": images,
                        "slash_commands": cmds,
                        "slash_args": [a.strip()[:200] for a in args],
                        "urls": URL.findall(clean)[:10],
                        "domains": sorted({d for d in DOMAIN.findall(clean) if re.fullmatch(r"[\w.:-]+", d)}),
                        "file_refs": sorted({x.lower() for x in EXT.findall(clean)}),
                        "text": clean[:TEXT_CAP],
                    }
                )
            elif o.get("type") == "assistant" and isinstance(content, list):
                for b in content:
                    if b.get("type") != "tool_use":
                        continue
                    name = b.get("name", "?")
                    inp = b.get("input") or {}
                    tools[name] += 1
                    if name.startswith("mcp__"):
                        mcp[name.split("__")[1]] += 1
                    elif name == "Skill" and inp.get("skill"):
                        skills[inp["skill"]] += 1
                    elif name in ("Bash", "PowerShell"):
                        sig = cli_signal(inp.get("command"))
                        if sig:
                            cli[sig] += 1
    return prompts, tools, mcp, skills, cli, meta


def est_tokens(s):
    # Measured: this JSONL tokenizes at ~2 characters per token (Read caps at 25k tokens).
    return len(s) // 2 + 1


LINE_MAX = 1900  # Read truncates lines longer than 2000 characters


def line_of(obj):
    """Serialize one JSONL record, shrinking it so the line stays readable in full."""
    if obj.get("urls"):
        obj["urls"] = [u[:200] for u in obj["urls"][:5]]
    line = json.dumps(obj, ensure_ascii=False)
    for field in ("text", "cmd", "slash_args", "urls"):
        while len(line) > LINE_MAX and obj.get(field):
            v = obj[field]
            obj[field] = v[: max(0, len(v) - (len(line) - LINE_MAX) - 20)] if isinstance(v, str) else v[:-1]
            if field == "text":
                obj["truncated"] = True
            line = json.dumps(obj, ensure_ascii=False)
    return line


def chunked(items, size):
    return [items[i:i + size] for i in range(0, len(items), size)] or [[]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=20)
    ap.add_argument("--root", default=os.path.join(
        os.environ.get("CLAUDE_CONFIG_DIR") or os.path.expanduser("~/.claude"), "projects"))
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--shard-tokens", type=int, default=20000,
                    help="estimated token budget per shard; keep under the reader's single-Read limit")
    ap.add_argument("--min-prompts", type=int, default=2, help="skip near-empty sessions")
    ap.add_argument("--exclude-session", action="append", default=[],
                    help="session id to skip (e.g. the session running this analysis)")
    a = ap.parse_args()

    files = [
        p for p in glob.glob(os.path.join(a.root, "*", "*.jsonl"))
        if os.path.splitext(os.path.basename(p))[0] not in a.exclude_session
    ]
    files.sort(key=os.path.getmtime, reverse=True)
    os.makedirs(a.out, exist_ok=True)

    sessions, raw_bytes = [], 0
    for path in files:
        if len(sessions) >= a.sessions:
            break
        prompts, tools, mcp, skills, cli, meta = scan(path)
        if len(prompts) < a.min_prompts:
            continue
        raw_bytes += os.path.getsize(path)
        sessions.append(
            {
                "session_id": os.path.splitext(os.path.basename(path))[0],
                "project": os.path.basename(os.path.dirname(path)),
                **meta,
                "prompt_count": len(prompts),
                "images_total": sum(p["images"] for p in prompts),
                "pasted_blocks": sum(p["pasted_block"] for p in prompts),
                "tools_used": dict(tools.most_common()),
                "mcp_servers_used": dict(mcp.most_common()),
                "skills_used": dict(skills.most_common()),
                "cli_commands": [[c, n] for c, n in cli.most_common()],
                "prompts": prompts,
            }
        )

    # Pack JSONL lines into shards under the token budget. A session larger than
    # one shard continues in the next; its header line is repeated so every shard
    # is self-describing. Readers cite full session ids, so the merge step can
    # count distinct sessions across shards without double counting.
    shards, cur, cur_tok = [], [], 0

    def flush():
        nonlocal cur, cur_tok
        if cur:
            shards.append(cur)
        cur, cur_tok = [], 0

    def add(line, hline=None):
        nonlocal cur_tok
        if cur and cur_tok + est_tokens(line) > a.shard_tokens:
            flush()
            if hline and line != hline:
                cur.append(hline)
                cur_tok += est_tokens(hline)
        cur.append(line)
        cur_tok += est_tokens(line)

    for s in sessions:
        sid = s["session_id"]
        header = {k: s[k] for k in ("session_id", "project", "cwd", "branch", "first_ts", "last_ts",
                                    "prompt_count", "images_total", "pasted_blocks",
                                    "mcp_servers_used", "skills_used")}
        hline = line_of({"type": "session", **header})
        add(hline)
        for chunk in chunked(list(s["tools_used"].items()), 30):
            add(line_of({"type": "tools", "session_id": sid, "tools_used": dict(chunk)}), hline)
        for cmd, n in s["cli_commands"][:60]:
            add(line_of({"type": "cli", "session_id": sid, "n": n, "cmd": cmd}), hline)
        for p in s["prompts"]:
            add(line_of({"type": "prompt", "session_id": sid, **p}), hline)
    flush()

    shard_files, extracted_bytes = [], 0
    for i, lines in enumerate(shards, 1):
        p = os.path.join(a.out, f"shard-{i}.jsonl")
        with open(p, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(lines))
            fh.write("\n" + json.dumps({"type": "end", "shard": i, "lines": len(lines)}) + "\n")
        shard_files.append(p)
        extracted_bytes += os.path.getsize(p)

    def total(key_fn):
        return dict(Counter(k for s in sessions for k in key_fn(s)).most_common())

    summary = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "sessions": len(sessions),
        "date_range": [min((s["first_ts"] for s in sessions if s["first_ts"]), default=None),
                       max((s["last_ts"] for s in sessions if s["last_ts"]), default=None)],
        "prompts": sum(s["prompt_count"] for s in sessions),
        "images": sum(s["images_total"] for s in sessions),
        "pasted_blocks": sum(s["pasted_blocks"] for s in sessions),
        "raw_mb": round(raw_bytes / 1e6, 1),
        "extracted_kb": round(extracted_bytes / 1e3, 1),
        "est_tokens": extracted_bytes // 2,
        "reduction": f"{(1 - extracted_bytes / raw_bytes) * 100:.1f}%" if raw_bytes else "n/a",
        "mcp_servers_total": dict(sum((Counter(s["mcp_servers_used"]) for s in sessions), Counter()).most_common()),
        "skills_total": dict(sum((Counter(s["skills_used"]) for s in sessions), Counter()).most_common()),
        "slash_commands_total": total(lambda s: [c for p in s["prompts"] for c in p["slash_commands"]]),
        "domains_total": total(lambda s: [d for p in s["prompts"] for d in p["domains"]]),
        "file_refs_total": total(lambda s: [f for p in s["prompts"] for f in p["file_refs"]]),
        "shards": len(shard_files),
        "shard_files": shard_files,
    }
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

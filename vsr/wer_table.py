"""Corpus WER (with substitutions/deletions/insertions) for each eval_adapter.py jsonl."""
import glob, json, re, sys
import jiwer

fix = lambda s: re.sub(r"(\w) '(\w)", r"\1'\2", s)
out = {}
for f in sorted(sys.argv[1:] or glob.glob("runs/results/*.jsonl")):
    r = [json.loads(l) for l in open(f)]
    refs = [fix(x["ref"]) for x in r]; hyps = [fix(x["hyp"]) for x in r]
    keep = [i for i, x in enumerate(refs) if x.strip()]
    m = jiwer.process_words([refs[i] for i in keep], [hyps[i] for i in keep])
    out[f.split("/")[-1][:-6]] = dict(clips=len(keep), ref_words=sum(len(refs[i].split()) for i in keep),
        wer=round(m.wer, 3), sub=m.substitutions, dele=m.deletions, ins=m.insertions)
print(json.dumps(out, indent=1))

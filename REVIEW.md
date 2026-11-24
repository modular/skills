# Skill review

When reviewing a skill, flag every violation of these rules, citing the
rule.

## 1. Review the entire skill, not only the diff

This is the most important rule. Review the whole skill as it reads after
the PR's additions: its `SKILL.md` and every file under it that the change
touches or that the changed text refers to. We've often found that a diff
reads well on its own, but read as a whole, the skill contradicts itself or
becomes inconsistent. Flag such cases.

## 2. Treat `oss/skills/` as public

Flag internal links (Slack, Notion, Linear, Google Docs, internal GitHub
URLs), partner or customer names, internal tickets, internal hosts and
remote GPU configs in `oss/skills/`. Flag a new OSS skill missing from
`oss/skills/README.md` or `oss/skills/.claude-plugin/marketplace.json`.

## 3. Do not copy-paste information, but point to where it lives

The repo changes quickly, so information hard-coded into a skill soon goes
stale. Flag text that restates what another skill, script, doc or code already
says, including a section that repeats a file it links to. Flag a snapshot
of the repo that will go stale: a `file:line` reference, or a copied list
of what a script does or what the code currently looks like. Link to the
source instead.

## 4. Prefer fewer instructions

Flag an addition that makes the skill more prescriptive without naming the
failure it fixes: detailed procedure for something an agent already does
well, required fields or formats, an extra README, or a plan or record no
step reads. Ask what went wrong without it. A skill states the few things
that matter, not how to do everything.

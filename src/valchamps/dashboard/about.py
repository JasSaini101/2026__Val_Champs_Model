"""Text of the dashboard's "How it works" tab: the data and what the model looks at."""

from __future__ import annotations

RESULTS_URL = "https://github.com/JasSaini101/2026__Val_Champs_Model#results"

DATA = """
Every tier-1 VALORANT match of 2025 and 2026, scraped from [vlr.gg](https://www.vlr.gg):

- **VCT regional leagues**: Kickoff, Stage 1 and Stage 2 in Americas, EMEA, Pacific and China,
  for both years. They show how teams rank within their region.
- **International events**: Masters and Champions. They are the only matches between regions,
  so they show how the regions compare.
- **Champions 2026 itself**: each finished match is added before the next daily update.

For each match: the teams and lineups, the map veto, and every map's score, halves and
round-by-round results, plus per-player stats.
"""

FEATURES = """
The model predicts **who wins each map**. Every input is computed only from matches played
before that map, so it never sees the future. For each of the two teams:

**Ratings**
- Team Elo rating, updated after every map and weighted by the round margin
- The team's Elo adjustment on this specific map
- A region strength offset, moved only by international matches

**Recent form**
- Map win rate and round difference over the last 5 matches
- Days of rest, and matches played in the last 30 days
- Experience: maps played overall and at international events
- Win rate at international events
- Head-to-head record against this opponent

**Map pool and veto**
- Win rate on this map, pulled toward the team's overall win rate when it has few games there
- How often the team picks and bans this map
- Whose pick the map is: the team's own, the opponent's, or the decider

**Roster**
- Share of the lineup unchanged from the previous match
- The lineup's average player rating over recent matches

**Standings**
- Finish in the team's latest league event, and whether it won it
- Circuit points earned this season
- Finish at the team's latest international event

**Match context**
- Which map is being played
- Whether it's an international event, and whether the teams are from different regions
"""

PIPELINE = """
1. **Map odds**: a neural network turns those inputs into the chance each team wins each map.
2. **Series odds**: the map veto is simulated from each team's pick and ban habits. Every
   possible veto is weighed by its probability, which gives exact Bo3 and Bo5 odds.
3. **Title odds**: the whole bracket is played out 100,000 times from the results so far.
4. **Daily update**: around noon ET, new results are pulled in, every rating and form number is
   brought up to date, and the bracket is simulated again.
"""

ACCURACY = f"""
Tested on events the model never saw during training, its probabilities are well calibrated:
when it says 60%, the team wins about 60% of the time. It does best on cross-region maps, the
matches that decide Champions. Pro VALORANT is close to a coin flip, though. Over a full series
it only matches a plain Elo rating. [Full results]({RESULTS_URL})
"""

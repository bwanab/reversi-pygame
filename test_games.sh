#!/usr/bin/env sh

python test_play.py --time_limit 600000 --rounds 100 --agent2 bwanab_2023.PositionHeuristicAgentPlusMinMoves --agent1 bwanab_2023.SimpleHeuristicAgent
python test_play.py --time_limit 600000 --rounds 100 --agent1 bwanab_2023.PositionHeuristicAgentPlusMinMoves --agent2 bwanab_2023.SimpleHeuristicAgent
python test_play.py --time_limit 600000 --rounds 100 --agent2 bwanab_2023.PositionHeuristicAgentPlusMinMoves --agent1 bwanab_2023.BBAgent
python test_play.py --time_limit 600000 --rounds 100 --agent1 bwanab_2023.PositionHeuristicAgentPlusMinMoves --agent2 bwanab_2023.BBAgent
python test_play.py --time_limit 600000 --rounds 100 --agent2 bwanab_2023.BBAgent --agent1 bwanab_2023.SimpleHeuristicAgent
python test_play.py --time_limit 600000 --rounds 100 --agent1 bwanab_2023.BBAgent --agent2 bwanab_2023.SimpleHeuristicAgent

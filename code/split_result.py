# coding: utf-8
import pandas as pd
dnn = pd.read_csv('result/dnn_0530131412_0.1847.csv')
board_session = set(pd.read_csv('../data/test_leaderboard_sessions.csv')['session_id'])
final_session = set(pd.read_csv('../data/test_final_sessions.csv')['session_id'])
test_leaderboard = dnn[dnn['session_id'].isin(board_session)]
final_leaderboard = dnn[dnn['session_id'].isin(final_session)]
test_leaderboard.to_csv('result/leaderboard_dnn.csv', index=False)
final_leaderboard.to_csv('result/final_dnn.csv', index=False)

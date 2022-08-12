# -*- coding:utf-8 -*-

#========================================================================
# Author: doubleQ
# File Name: 1_generate_candidate.py
# Created Date: 2022-07-12
# Description:
# =======================================================================

import pandas as pd
from argparse import ArgumentParser
from datetime import datetime
version = datetime.now().strftime("%m%d%H%M%S")

parser = ArgumentParser()
parser.add_argument("--files", nargs="+", default=[])
args = parser.parse_args()
print(args.files)
ans = None
for f in args.files:
    df = pd.read_csv(f)
    if ans is None:
        ans = df
    else:
        #ans['pred'] = ans['pred'] + df['pred']
        ans = ans.merge(df, on=['session_id', 'item_id'], how='left').fillna(0)
        ans['pred'] = ans['pred_x'] + ans['pred_y']
        ans = ans.drop(columns=['pred_x', 'pred_y'])

ans = ans.sort_values(by=['session_id','pred'], ascending=False).groupby('session_id')['item_id'].apply(lambda x: list(x)[:100]).reset_index()
ans.columns = ['session_id', 'item_id']
ans = ans.explode('item_id')
ans['rank'] = list(range(1, 101)) * len(set(ans['session_id']))
board_session = set(pd.read_csv('../data/test_leaderboard_sessions.csv')['session_id'])
final_session = set(pd.read_csv('../data/test_final_sessions.csv')['session_id'])
test_leaderboard = ans[ans['session_id'].isin(board_session)]
final_leaderboard = ans[ans['session_id'].isin(final_session)]
test_leaderboard.to_csv('result/submit_leaderboard_{}.csv'.format(version), index=False)
final_leaderboard.to_csv('result/submit_final_{}.csv'.format(version), index=False)

from bs4 import BeautifulSoup
import pandas as pd
import requests
import numpy as np
import datetime as dt
from geopy import distance
import matplotlib.pyplot as plt

# load csv of nba team's city airport coordinates
coords = pd.read_csv('./Final/nba_team_locations.csv', index_col='Team')


def getDistance(team1, team2):
    ''' Calculates the distance between two teams' cities
        inputs: home team of previous game, home team of current game
        output: distance traveled from last game to current game, in miles
    '''
    dist = distance.distance(coords[['Latitude', 'Longitude']].loc[team1],
                             coords[['Latitude', 'Longitude']].loc[team2]).miles
    return dist


# team dictionary for url request
team_dict = {'Hawks': 'ATL', 'Celtics': 'BOS', 'Nets': 'BRK', 'Hornets': 'CHO', 'Bulls': 'CHI',
             'Cavaliers': 'CLE', 'Mavericks': 'DAL', 'Nuggets': 'DEN', 'Pistons': 'DET', 'Warriors': 'GSW',
             'Rockets': 'HOU', 'Pacers': 'IND', 'Clippers': 'LAC', 'Lakers': 'LAL', 'Grizzlies': 'MEM',
             'Heat': 'MIA', 'Bucks': 'MIL', 'Timberwolves': 'MIN', 'Pelicans': 'NOP', 'Knicks': 'NYK',
             'Thunder': 'OKC', 'Magic': 'ORL', '76ers': 'PHI', 'Suns': 'PHO', 'Trailblazers': 'POR',
             'Kings': 'SAC', 'Spurs': 'SAS', 'Raptors': 'TOR', 'Jazz': 'UTA', 'Wizards': 'WAS'}

# search for team you want stats for
team_in = input(f'Enter team: ')
print()

team = team_dict[team_in.title()]  # get team abbreviation from dictionary

# scrape basketball-reference.com for team game log table
url = f'https://www.basketball-reference.com/teams/{team}/2022/gamelog/'
res = requests.get(url)
soup = BeautifulSoup(res.content, 'html.parser')
table = soup.find(id='tgl_basic')

# reads scraped html and makes data frame
tdata = pd.read_html(str(table))[0]
df = pd.DataFrame(tdata)
df = df.droplevel(0, 'columns')

# clean up data in data frame
df.dropna(subset=['G'], inplace=True)
df.index = df['G']
df.drop(['Rk', 'G'], 'columns', inplace=True)
df.drop('G', inplace=True)
df.rename(columns={'Unnamed: 3_level_1': 'Home/Away'}, inplace=True)
df['Home/Away'] = df['Home/Away'].fillna('vs')
df['Date'] = pd.to_datetime(df['Date'])
opp_colnames = {'Opp': ['Opp', 'OppScore'], 'Tm': ['TeamScore']}
df.rename(columns=lambda col: opp_colnames[col].pop(
    0) if col in opp_colnames.keys() else col, inplace=True)

# create separate dataframes for team's offensive and defensive stats
off_df = df.iloc[:, :22]
def_df = df.iloc[:, 23:]
def_df = def_df.add_prefix('Opp_')

# merge stats into one dataframe
df = off_df.merge(def_df, on='G', how='left')

# convert columns to appropriate types for calculations
ncol = ['TeamScore', 'OppScore',
        'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%',
        'ORB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF',
        'Opp_FG', 'Opp_FGA', 'Opp_FG%', 'Opp_3P', 'Opp_3PA', 'Opp_3P%',
        'Opp_FT', 'Opp_FTA', 'Opp_FT%', 'Opp_ORB', 'Opp_TRB', 'Opp_AST',
        'Opp_STL', 'Opp_BLK', 'Opp_TOV', 'Opp_PF']
df[ncol] = df[ncol].apply(pd.to_numeric)

df['Diff'] = df['TeamScore']-df['OppScore']

# calculate offensive and defensive four factors stats
df['eFG%'] = (df['FG'] + 0.5 * df['3P'])/df['FGA']
df['TOV%'] = df['TOV'] / (df['FGA'] + 0.44 * df['FTA'] + df['TOV'])
df['FT Rate'] = df['FT'] / df['FTA']
df['ORB%'] = df['ORB'] / (df['ORB'] + (df['Opp_TRB']-df['Opp_ORB']))

df['Opp eFG%'] = (df['Opp_FG'] + 0.5 * df['Opp_3P'])/df['Opp_FGA']
df['Opp TOV%'] = (df['Opp_TOV'] / (df['Opp_FGA'] +
                  0.44 * df['Opp_FTA'] + df['Opp_TOV']))
df['Opp FT Rate'] = df['Opp_FT'] / df['Opp_FTA']
df['DRB%'] = (df['TRB']-df['ORB']) / ((df['TRB']-df['ORB']) + df['Opp_ORB'])

# get distance traveled for each game
away = df['Home/Away'] == '@'
home = df['Home/Away'] == 'vs'
for i in df[away].index:
    prevGame = str(int(i)-1)
    if prevGame == '0':
        df.loc[i, 'Miles Traveled'] = 0
        continue

    # away game after home game
    if df.loc[prevGame, 'Home/Away'] == 'vs' and prevGame != '0':
        df.loc[i, 'Miles Traveled'] = getDistance(team, df.loc[i][2])
    # away game after away game
    elif df.loc[prevGame, 'Home/Away'] == '@' and prevGame != '0':
        df.loc[i, 'Miles Traveled'] = getDistance(
            df.loc[prevGame][2], df.loc[i][2])
for i in df[home].index:
    prevGame = str(int(i)-1)
    if prevGame == '0':
        df.loc[i, 'Miles Traveled'] = 0
        continue

    # home game after home game
    if df.loc[prevGame, 'Home/Away'] == 'vs' and prevGame != '0':
        df.loc[i, 'Miles Traveled'] = getDistance(team, team)
    # home game after away game
    elif df.loc[prevGame, 'Home/Away'] == '@' and prevGame != '0':
        df.loc[i, 'Miles Traveled'] = getDistance(df.loc[prevGame][2], team)

df['Miles Traveled'] = df['Miles Traveled'].apply(np.ceil).astype('int64')

#  calculates days of rest between each game and enters it in new column
df['Days of Rest'] = np.nan
for i in range(1, len(df['Date'])):
    df.loc[str(i+1), 'Days of Rest'] = df.iloc[i, 0] - df.iloc[i-1, 0]
df.loc['1', 'Days of Rest'] = 3
df['Days of Rest'] = pd.to_timedelta(
    df['Days of Rest'], unit='d')

# separates rest days and miles traveled into bins
df['Days of Rest'] = pd.cut(df['Days of Rest'], bins=pd.to_timedelta([0, 2, 3, 28], unit='d'),
                            labels=['0 days', '1 day', '2+ days'], right=False)

df['Miles Traveled'] = pd.cut(df['Miles Traveled'], bins=[0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500],
                              labels=['0-250', '250-500', '500-750', '750-1000',
                                      '1000-1250', '1250-1500', '1500-1750', '1750-2000',
                                      '2000-2250', '2250-2500'], include_lowest=True, right=False)

# dataframe for only four factors stats
ff_df = pd.DataFrame(df[['Date', 'Home/Away', 'Opp', 'W/L', 'Diff',
                        'eFG%', 'TOV%', 'FT Rate', 'ORB%',
                         'Opp eFG%', 'Opp TOV%', 'Opp FT Rate', 'DRB%',
                         'Miles Traveled', 'Days of Rest']], index=df.index)

# four factors vs miles traveled
ff_miles = ff_df.groupby(['Miles Traveled'])[
    ['eFG%', 'TOV%', 'FT Rate', 'ORB%', 'Opp eFG%', 'Opp TOV%', 'Opp FT Rate', 'DRB%']].mean()
ff_miles.dropna(inplace=True)
print(f'Four Factors vs Miles Traveled DF\n{ff_miles}\n')

# four factors vs rest days
ff_rest = ff_df.groupby(['Days of Rest'])[
    ['eFG%', 'TOV%', 'FT Rate', 'ORB%', 'Opp eFG%', 'Opp TOV%', 'Opp FT Rate', 'DRB%']].mean()
print(f'Four Factors vs Rest Days DF\n{ff_rest}\n')

# visualizations – four factors vs rest days
fig1 = plt.figure(figsize=(24, 16))

efg_r = fig1.add_subplot(2, 2, 1)
tov_r = fig1.add_subplot(2, 2, 2)
reb_r = fig1.add_subplot(2, 2, 3)
ft_r = fig1.add_subplot(2, 2, 4)

efg_r.plot(ff_rest.index, ff_rest['eFG%'], label='eFG%')
efg_r.plot(ff_rest.index, ff_rest['Opp eFG%'], label='Opp eFG%')

tov_r.plot(ff_rest.index, ff_rest['TOV%'], label='TOV%')
tov_r.plot(ff_rest.index, ff_rest['Opp TOV%'], label='Opp TOV%')

reb_r.plot(ff_rest.index, ff_rest['ORB%'], label='ORB%')
reb_r.plot(ff_rest.index, ff_rest['DRB%'], label='DRB%')

ft_r.plot(ff_rest.index, ff_rest['FT Rate'], label='FT Rate')
ft_r.plot(ff_rest.index, ff_rest['Opp FT Rate'], label='Opp FT Rate')

efg_r.set_xlabel('Days of Rest')
efg_r.set_ylabel('eFG%')
efg_r.set_title('eFG% vs Days of Rest Entering Game')
efg_r.legend(loc='best')

tov_r.set_xlabel('Days of Rest')
tov_r.set_ylabel('TOV%')
tov_r.set_title('TOV% vs Days of Rest Entering Game')
tov_r.legend(loc='best')

reb_r.set_xlabel('Days of Rest')
reb_r.set_ylabel('ORB%')
reb_r.set_title('ORB% vs Days of Rest Entering Game')
reb_r.legend(loc='best')

ft_r.set_xlabel('Days of Rest')
ft_r.set_ylabel('FT Rate')
ft_r.set_title('FT Rate vs Days of Rest Entering Game')
ft_r.legend(loc='best')


# write to png files for searched team
plt.savefig(f'./Final/{team_dict[team_in.title()]}_RestDays.png')


# visualizations – four factors vs miles traveled
fig2 = plt.figure(figsize=(24, 16))

efg_mi = fig2.add_subplot(2, 2, 1)
tov_mi = fig2.add_subplot(2, 2, 2)
reb_mi = fig2.add_subplot(2, 2, 3)
ft_mi = fig2.add_subplot(2, 2, 4)

efg_mi.plot(ff_miles.index, ff_miles['eFG%'], label='eFG%')
efg_mi.plot(ff_miles.index, ff_miles['Opp eFG%'], label='Opp eFG%')

tov_mi.plot(ff_miles.index, ff_miles['TOV%'], label='TOV%')
tov_mi.plot(ff_miles.index, ff_miles['Opp TOV%'], label='Opp TOV%')

reb_mi.plot(ff_miles.index, ff_miles['ORB%'], label='ORB%')
reb_mi.plot(ff_miles.index, ff_miles['DRB%'], label='DRB%')

ft_mi.plot(ff_miles.index, ff_miles['FT Rate'], label='FT Rate')
ft_mi.plot(ff_miles.index, ff_miles['Opp FT Rate'], label='Opp FT Rate')

efg_mi.set_xlabel('Miles Traveled')
efg_mi.set_ylabel('eFG%')
efg_mi.set_title('eFG% vs Miles Traveled Entering Game')
efg_mi.legend(loc='best')

tov_mi.set_xlabel('Miles Traveled')
tov_mi.set_ylabel('TOV%')
tov_mi.set_title('TOV% vs Miles Traveled Entering Game')
tov_mi.legend(loc='best')

reb_mi.set_xlabel('Miles Traveled')
reb_mi.set_ylabel('ORB%')
reb_mi.set_title('ORB% vs Miles Traveled Entering Game')
reb_mi.legend(loc='best')

ft_mi.set_xlabel('Miles Traveled')
ft_mi.set_ylabel('FT Rate')
ft_mi.set_title('FT Rate vs Miles Traveled Entering Game')
ft_mi.legend(loc='best')

# write to png files for searched team
plt.savefig(f'./Final/{team_dict[team_in.title()]}_MilesTraveled.png')

# show figures
plt.show()

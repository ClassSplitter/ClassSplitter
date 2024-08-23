import os
import subprocess
from typing import List
from dateutil import parser
from tqdm import tqdm
from god_class import GodClass, DictHandler
import numpy as np

dict_handler1 = DictHandler('position_data.csv')
good_classes1 = dict_handler1.get_classes()

dict_handler2 = DictHandler('GodClassRefactorDictionary.csv')
good_classes2: List[GodClass] = dict_handler2.get_classes()

good_classes: List[GodClass] = good_classes1 + good_classes2

print(f"total class number: {len(good_classes2)}")

num_of_developers = []
num_of_commits = []
years_of_histories = []

for c in tqdm(good_classes):
    try:
        project_path = c.data.at[c.data.iloc[0].name, 'project path']
        os.chdir(project_path)
        first_commit_hash = subprocess.run('git rev-list --max-parents=0 HEAD', shell=True, capture_output=True, text=True, encoding='utf-8').stdout
        first_commit_date_str = subprocess.run(f'git show {first_commit_hash}', shell=True, capture_output=True, text=True, encoding='utf-8').stdout.split('\n')[2].split('Date:')[1].strip()

        last_commit_date_str = subprocess.run('git log -1 --pretty="%cd"', shell=True, capture_output=True, text=True, encoding='utf-8').stdout

        first_commit_date = parser.parse(first_commit_date_str)
        last_commit_date = parser.parse(last_commit_date_str)

        diff_time = last_commit_date - first_commit_date
        project_history_time = diff_time.days / 365.25

        output = subprocess.run('git shortlog -s -n', shell=True, capture_output=True, text=True, encoding='utf-8')
        lines = output.stdout.splitlines()
        commits = []
        for line in lines:
            num = int(line.split('\t')[0].strip())
            commits.append(num)
        
        num_of_developers.append(len(lines))
        num_of_commits.append(sum(commits))
        years_of_histories.append(project_history_time)
    except UnicodeDecodeError as ude:
        pass
    except AttributeError as ae:
        pass


print(f"num of developers: {min(num_of_developers)} to {max(num_of_developers)}, average: {sum(num_of_developers) / len(num_of_developers)}, median: {np.median(num_of_developers)}")
print(f"num of commits: {min(num_of_commits)} to {max(num_of_commits)}, average: {sum(num_of_commits) / len(num_of_commits)}, median: {np.median(num_of_commits)}")
print(f"years of histories: {min(years_of_histories)} to {max(years_of_histories)}, average: {sum(years_of_histories) / len(years_of_histories)}, median: {np.median(years_of_histories)}")

print(num_of_developers)

print(num_of_commits)

print(years_of_histories)






# os.chdir('D:\\Top1kProjects\\2dxgujun_AndroidTagGroup')

# first_commit_hash = subprocess.run('git rev-list --max-parents=0 HEAD', shell=True, capture_output=True, text=True).stdout
# first_commit_date_str = subprocess.run(f'git show {first_commit_hash}', shell=True, capture_output=True, text=True).stdout.split('\n')[2].split('Date:')[1].strip()

# last_commit_date_str = subprocess.run('git log -1 --pretty="%cd"', shell=True, capture_output=True, text=True).stdout

# first_commit_date = parser.parse(first_commit_date_str)
# last_commit_date = parser.parse(last_commit_date_str)

# diff_time = last_commit_date - first_commit_date
# project_history_time = diff_time.days / 365.25

# output = subprocess.run('git shortlog -s -n', shell=True, capture_output=True, text=True)
# lines = output.stdout.splitlines()
# commits = []
# for line in lines:
#     num = int(line.split('\t')[0].strip())
#     commits.append(num)

# print(output)

# print(f"lines=contributors = {len(lines)}")

# print(f"commit numbers = {sum(commits)}")

# print(f"first commit date = {first_commit_date}")

# print(f" last commit date = {last_commit_date}")

# print(f"project history : {project_history_time}")

import multiprocessing
import pickle
import random
import time
import datetime
import numpy as np

from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


def get_client():
    client = Elasticsearch([{'host':'10.0.0.200', 'port':9200, 'timeout':300}])

    if client is not None:
        return client
    else:
        raise Exception('Connection could not be established.')


def get_fork_events_between(start_date, end_date):
    client = get_client()
    s = Search(using=client, index="events") \
        .query("match", type="ForkEvent") \
        .filter('range', created_at={'gte': start_date, 'lte': end_date}) \
        .sort("created_at")
    response = s.scan()
    return response


def get_repos_forked_between(start_date, end_date):
    client = get_client()
    s = Search(using=client, index="events") \
        .query("match", type="ForkEvent") \
        .filter('range', created_at={'gte': start_date, 'lte': end_date}) \
        .sort("created_at")
    response = s.scan()
    repos = [hit.to_dict()['repo']['id_h'] for hit in response]
    return repos


def get_repos_forkcounts_created_between(start_date, end_date):
    client = get_client()
    s = Search(using=client, index="repos-upd") \
        .filter('range', created_at={'gte': start_date, 'lte': end_date}) \
        .sort("created_at")
    response = s.scan()
    hits = {}
    for i, hit in enumerate(response):
        print(i)
        hits[hit.to_dict()['ght_id_h']] = hit.to_dict()['forks_count']
    pickle.dump(hits, open('fork_counts_filtered.pkl', 'wb'))
    return hits


def get_repos_with_forkcount_between(min_count, max_count):
    client = get_client()
    s = Search(using=client, index="repos-upd") \
        .filter('range', forks_count={'gte': min_count, 'lte': max_count})
    response = s.scan()
    hits = [hit.to_dict()['ght_id_h'] for hit in response]
    return hits


def get_user_profile(login_id):
    client = get_client()
    s = Search(using=client, index="users") \
        .query("match", login_h__keyword=login_id)
    response = s.execute()
    hits = [hit.to_dict() for hit in response]
    assert len(hits) == 1
    return hits[0]


def get_repo_profile(repo_id):
    client = get_client()
    s = Search(using=client, index="repos-upd") \
        .query("match", ght_id_h__keyword=repo_id)
    response = s.execute()
    hits = [hit.to_dict() for hit in response]
    assert len(hits) == 1
    return hits[0]


def get_follower_list(login_id):
    client = get_client()
    s = Search(using=client, index="followers") \
        .query("match", login_h__keyword=login_id)

    response = s.execute()
    followers_list =[]
    for hit in response:
        followers = hit.to_dict()['followers']
        for follower in followers:
            followers_list.append(follower['login_h'])
    return followers_list


def is_follower(login_id_1, login_id_2):
    followers = get_follower_list(login_id_1)
    return True if login_id_2 in followers else False


def get_forked_repos(login_id, start_date, end_date):
    client = get_client()
    s = Search(using=client, index="events") \
        .query("match", type="ForkEvent") \
        .query("match", actor__login_h=login_id) \
        .filter('range', created_at={'gte': start_date, 'lte': end_date})
    response = s.scan()
    return [hit.repo.id_h for hit in response]


def did_forked(login_id, repo_id, start_date, end_date):
    client = get_client()
    s = Search(using=client, index="events") \
        .query("match", type="ForkEvent") \
        .query("match", actor__login_h=login_id) \
        .query("match", repo__id_h=repo_id) \
        .filter('range', created_at={'gte': start_date, 'lte': end_date})
    response = s.execute()
    hits = [hit.to_dict() for hit in response]
    if len(hits) == 1:
        return True
    return False


def is_watching(login_id, repo_id):
    client = get_client()
    s = Search(using=client, index="events") \
        .query("match", type="WatchEvent") \
        .query("match", actor__login_h=login_id) \
        .query("match", repo__id_h=repo_id)
    response = s.execute()
    hits = [hit.to_dict() for hit in response]
    if len(hits) > 0:
        return True
    return False


def get_fork_events(followers, repos, start_date, end_date):
    client = get_client()
    s = Search(using=client, index="events") \
        .query("match", type="ForkEvent") \
        .filter('range', created_at={'gte': start_date, 'lte': end_date}) \
        .filter('terms', actor__login_h__keyword=followers) \
        .filter('terms', repo__id_h__keyword=repos)
    response = s.scan()
    return response


def get_follower_forks(user_login_id, start_date, end_date, output):
    repos = get_forked_repos(user_login_id, start_date, end_date)
    followers = get_follower_list(user_login_id)
    events = get_fork_events(followers, repos, start_date, end_date)
    fork_events = []
    for event in events:
        event = event.to_dict()
        fork_events.append((user_login_id, event['repo']['id_h'], event['actor']['login_h'], 'True'))
    output.put(fork_events)


def get_forked_users(repo_id, start_date, end_date):
    client = get_client()
    s = Search(using=client, index="events") \
        .query("match", type="ForkEvent") \
        .query("match", repo__id_h=repo_id) \
        .filter('range', created_at={'gte': start_date, 'lte': end_date})
    response = s.execute()
    hits = [hit.to_dict()['actor']['login_h'] for hit in response]
    return hits


def get_repositories_created_between(start_date, end_date):
    client = get_client()
    s = Search(using=client, index="events") \
        .query("match", type="CreateEvent") \
        .filter('range', created_at={'gte': start_date, 'lte': end_date}) \
        .sort("created_at")
    response = s.scan()
    repos = []
    for hit in response:
        repos.append(hit.to_dict()['repo']['id_h'])
    return repos


def get_repositories_created_between_filtered(start_date, end_date, repo_ids):
    client = get_client()
    s = Search(using=client, index="events") \
        .query("match", type="CreateEvent") \
        .filter('range', created_at={'gte': start_date, 'lte': end_date}) \
        .filter('terms', repo__id_h__keyword=repo_ids) \
        .sort("created_at")
    response = s.scan()
    repos = []
    for hit in response:
        repos.append(hit.to_dict()['repo']['id_h'])
    return repos


def get_repositories_created_between_filtered_type(start_date, end_date, repo_ids, type):
    client = get_client()
    s = Search(using=client, index="repos-upd") \
        .query("match", owner__type=type) \
        .filter("terms", ght_id_h__keyword=repo_ids) \
        .filter('range', created_at={'gte': start_date, 'lte': end_date}) \

    response = s.scan()
    repos = []
    for hit in response:
        repos.append(hit.to_dict()['ght_id_h'])
    return repos


def get_fork_events_for_repos_between(repo_ids, start_date, end_date):
    client = get_client()
    s = Search(using=client, index="events") \
        .query("match", type="ForkEvent") \
        .filter('terms', repo__id_h__keyword=repo_ids)\
        .filter('range', created_at={'gte': start_date, 'lte': end_date})
    response = s.execute()
    return response.hits.total


def get_fork_events_for_repo_between(repo_id, start_date, end_date):
    client = get_client()
    s = Search(using=client, index="events") \
        .query("match", type="ForkEvent") \
        .query("match", repo__id_h=repo_id) \
        .filter('range', created_at={'gte': start_date, 'lte': end_date})
    response = s.execute()
    return response.hits.total


def get_repo_create_events_between(start_date, end_date):
    client = get_client()
    s = Search(using=client, index="events") \
        .query("match", type="CreateEvent") \
        .query("match", payload__ref_type='repository') \
        .filter('range', created_at={'gte': start_date, 'lte': end_date})
    response = s.execute()
    return response.hits.total


def get_push_events_for_repo_between(repo_id, start_date, end_date):
    client = get_client()
    s = Search(using=client, index="events") \
        .query("match", type="PushEvent") \
        .query("match", repo__id_h=repo_id) \
        .filter('range', created_at={'gte': start_date, 'lte': end_date})
    response = s.execute()
    return response.hits.total


def get_push_events_for_repos_between(repo_ids, start_date, end_date):
    client = get_client()
    s = Search(using=client, index="events") \
        .query("match", type="PushEvent") \
        .filter('terms', repo__id_h__keyword=repo_ids)\
        .filter('range', created_at={'gte': start_date, 'lte': end_date})
    response = s.execute()
    return response.hits.total


def get_watch_events_for_repo_between(repo_id, start_date, end_date):
    client = get_client()
    s = Search(using=client, index="events") \
        .query("match", type="WatchEvent") \
        .query("match", repo__id_h=repo_id) \
        .filter('range', created_at={'gte': start_date, 'lte': end_date})
    response = s.execute()
    return response.hits.total


def get_watch_events_for_repos_between(repo_ids, start_date, end_date):
    client = get_client()
    s = Search(using=client, index="events") \
        .query("match", type="WatchEvent") \
        .filter('terms', repo__id_h__keyword=repo_ids)\
        .filter('range', created_at={'gte': start_date, 'lte': end_date})
    response = s.execute()
    return response.hits.total


def get_issues_closed_for_repo_between(repo_id, start_date, end_date):
    client = get_client()
    s = Search(using=client, index="events") \
        .query("match", type="IssuesEvent") \
        .query("match", repo__id_h=repo_id) \
        .query("match", actions="closed") \
        .filter('range', created_at={'gte': start_date, 'lte': end_date})
    response = s.scan()
    count = 0
    for hit in response:
        count += 1
    return count


def get_fork_events_for_repo_after_creation(repo_ids, time_interval):
    forks_in_interval = {}
    for i, repo_id in enumerate(repo_ids):
        repo_create_time = get_repo_profile(repo_id)['created_at']
        start_date = datetime.datetime.strptime(repo_create_time, '%Y-%m-%dT%H:%M:%SZ')
        end_date = start_date + datetime.timedelta(days=time_interval)
        end_date = datetime.datetime.strftime(end_date, '%Y-%m-%dT%H:%M:%SZ')
        forks = get_fork_events_for_repo_between(repo_id, repo_create_time, end_date)
        forks_in_interval[repo_id] = forks
        print(i)
    pickle.dump(forks_in_interval, open('forks_in_' + str(time_interval) + '_days.pkl', 'wb'))


def plot_fork_events(time_interval):
    forks_in_interval = pickle.load(open('forks_in_' + str(time_interval) + '_days.pkl', 'rb'))
    counts = list(forks_in_interval.values())
    y, binEdges = np.histogram(counts, bins=5)
    bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
    plt.yscale('log')
    plt.xscale('log')
    plt.plot(bincenters, y, '-')
    plt.show()
    #plt.hist(counts, 5, log=True, histtype = 'step')
    #plt.show()

if __name__ == '__main__':
    # filtered based on repo created time - repos that are created after 01-01-2015
    repo_ids = pickle.load(open('top_0.1_repositories_filtered.pkl', 'rb'))

    # scenario 1
    # get_fork_events_for_repo_after_creation(repo_ids, 180)
    # plot_fork_events(180)

    # scenario 2
    # months = ['2015-01-01', '2015-02-01', '2015-03-01', '2015-04-01', '2015-05-01', '2015-06-01', '2015-07-01', '2015-08-01',
    #           '2015-09-01', '2015-10-01', '2015-11-01', '2015-12-01', '2016-01-01', '2016-02-01', '2016-03-01', '2016-04-01', '2016-05-01', '2016-06-01', '2016-07-01', '2016-08-01',
    #           '2016-09-01', '2016-10-01', '2016-11-01', '2016-12-01', '2017-01-01']
    #
    # fork_events, push_events, watch_events = {}, {}, {}
    # for i in range(12):
    #     start, end = months[i], months[i + 1]
    #     fork_events[i], push_events[i], watch_events[i] = [], [], []
    #     repos = get_repositories_created_between_filtered(start, end, repo_ids)
    #     for j in range(i+1, i+13):
    #         start, end = months[j], months[j+1]
    #         fork_events[i].append(get_fork_events_for_repos_between(repos, start, end)/len(repos))
    #         push_events[i].append(get_push_events_for_repos_between(repos, start, end)/len(repos))
    #         watch_events[i].append(get_watch_events_for_repos_between(repos, start, end)/len(repos))
    # fork_events = [fork_events[key] for key in fork_events.keys()]
    # push_events = [push_events[key] for key in push_events.keys()]
    # watch_events = [watch_events[key] for key in watch_events.keys()]
    # plt.plot(np.mean(fork_events, axis=1), label="fork events")
    # plt.plot(np.mean(push_events, axis=1), label="push events")
    # plt.plot(np.mean(watch_events, axis=1), label="watch events")
    # plt.xlabel("Months after creation")
    # plt.ylabel("Average # of events per month")
    # plt.legend()
    # plt.show()

    # scenario 4
    months = ['2015-01-01', '2015-02-01', '2015-03-01', '2015-04-01', '2015-05-01', '2015-06-01', '2015-07-01', '2015-08-01',
              '2015-09-01', '2015-10-01', '2015-11-01', '2015-12-01', '2016-01-01', '2016-02-01', '2016-03-01', '2016-04-01', '2016-05-01', '2016-06-01', '2016-07-01', '2016-08-01',
              '2016-09-01', '2016-10-01', '2016-11-01', '2016-12-01', '2017-01-01']

    fork_events_org, watch_events_org = {}, {}
    fork_events_user, watch_events_user = {}, {}

    for i in range(12):
        start, end = months[i], months[i + 1]
        fork_events_org[i], watch_events_org[i] = [], []
        fork_events_user[i], watch_events_user[i] = [], []
        repos_org = get_repositories_created_between_filtered_type(start, end, repo_ids, 'Organization')
        repos_user = get_repositories_created_between_filtered_type(start, end, repo_ids, 'User')
        for j in range(i+1, i+13):
            start, end = months[j], months[j+1]
            fork_events_org[i].append(get_fork_events_for_repos_between(repos_org, start, end))
            watch_events_org[i].append(get_watch_events_for_repos_between(repos_org, start, end))
            fork_events_user[i].append(get_fork_events_for_repos_between(repos_user, start, end))
            watch_events_user[i].append(get_watch_events_for_repos_between(repos_user, start, end))
    fork_events_org = [fork_events_org[key] for key in fork_events_org.keys()]
    watch_events_org = [watch_events_org[key] for key in watch_events_org.keys()]
    fork_events_user = [fork_events_user[key] for key in fork_events_user.keys()]
    watch_events_user = [watch_events_user[key] for key in watch_events_user.keys()]
    plt.plot(np.mean(fork_events_org, axis=0), label="fork events - organization")
    plt.plot(np.mean(watch_events_org, axis=0), label="watch events - organization")
    plt.plot(np.mean(fork_events_user, axis=0), label="fork events - user")
    plt.plot(np.mean(watch_events_user, axis=0), label="watch events - user")
    plt.xlabel("Months after creation")
    plt.ylabel("Average # of events per month")
    plt.legend()
    plt.show()
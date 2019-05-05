from utils.database_util import *
import networkx as nx
import os, datetime
import numpy as np
import pickle, time, multiprocessing
import random
import networkit as nkt


def build_graph(events, min_repo_fork_count, min_user_fork_count, start_date, end_date):
    repo_fork_count = {}
    user_fork_count = {}
    for event in events:
        user_id = event.split(',')[0]
        repo_id = event.split(',')[1]
        if repo_id in repo_fork_count:
            repo_fork_count[repo_id] += 1
        else:
            repo_fork_count[repo_id] = 1
        if user_id in user_fork_count:
            user_fork_count[user_id] += 1
        else:
            user_fork_count[user_id] = 1
    filtered_repos = set([repo_id for repo_id in repo_fork_count if repo_fork_count[repo_id] > min_repo_fork_count])
    filtered_users = set([user_id for user_id in user_fork_count if user_fork_count[user_id] > min_user_fork_count])

    print(len(filtered_repos), len(filtered_users))

    g = nx.DiGraph()
    users = set([])
    repos = set([])
    for event in events:
        user_id = event.split(',')[0]
        repo_id = event.split(',')[1]
        if repo_id in filtered_repos and user_id in filtered_users:
            users.add(user_id)
            repos.add(repo_id)
            g.add_edge(repo_id, user_id)

    print(len(users), len(repos), len(g.edges()))

    pickle.dump({'users': users, 'repos': repos}, open('node_groups_' + start_date + '_' + end_date + '.pkl', 'w'))

    for i, user in enumerate(users):
        followers = set(get_follower_list(user)).intersection(users)
        for follower in followers:
            g.add_edge(user, follower)
        print(i)

    nx.write_gpickle(g, 'graph_' + start_date + '_' + end_date + '.pkl')


def build_nkt_graph():
    g = nx.read_gpickle('/home/social-sim/PycharmProjects/Information_Diffusion/data/graph_2016-08-01_2017-01-01.pkl')
    node_groups = pickle.load(open('/home/social-sim/PycharmProjects/Information_Diffusion/data/node_groups_2016-08-01_2017-01-01.pkl', 'rb'))
    node_mapping = {}
    for i, node in enumerate(list(g)):
         node_mapping[node] = i
    num_nodes = len(node_mapping.keys())

    nkt_graph = nkt.graph.Graph(n=num_nodes)
    for edge in g.edges():
        nkt_graph.addEdge(node_mapping[edge[0]], node_mapping[edge[1]])
    return node_mapping, nkt_graph


def build_train_graph():
    events = []
    f = open('/home/social-sim/PycharmProjects/Information_Diffusion/EventLogs/ForkEventLog_2016-08-01_2016-09-01_20180511_001405.csv')
    header = f.readline()
    events.extend(f.readlines())

    f = open('/home/social-sim/PycharmProjects/Information_Diffusion/EventLogs/ForkEventLog_2016-09-01_2016-10-01_20180511_001732.csv')
    header = f.readline()
    events.extend(f.readlines())

    f = open('/home/social-sim/PycharmProjects/Information_Diffusion/EventLogs/ForkEventLog_2016-10-01_2016-11-01_20180511_002125.csv')
    header = f.readline()
    events.extend(f.readlines())

    f = open('/home/social-sim/PycharmProjects/Information_Diffusion/EventLogs/ForkEventLog_2016-11-01_2016-12-01_20180511_002552.csv')
    header = f.readline()
    events.extend(f.readlines())

    f = open('/home/social-sim/PycharmProjects/Information_Diffusion/EventLogs/ForkEventLog_2016-12-01_2017-01-01_20180511_003037.csv')
    header = f.readline()
    events.extend(f.readlines())

    print(len(events))

    build_graph(events, 25, 10, '2016-08-01', '2017-01-01')


def get_edges_from_testset():
    f = open('/home/social-sim/PycharmProjects/Information_Diffusion/EventLogsForkEventLog_2017-01-01_2017-02-01_20180511_003510.csv')
    header = f.readline()
    events = f.readlines()

    train_data = pickle.load(open('/home/social-sim/PycharmProjects/Information_Diffusion/data/node_groups_2016-08-01_2017-01-01.pkl', 'r'))
    repos = train_data['repos']
    users = train_data['users']

    edges = []
    for event in events:
        user_id = event.split(',')[0]
        repo_id = event.split(',')[1]
        if repo_id in repos and user_id in users:
            edges.append((repo_id, user_id))

    pickle.dump(edges, open('test_edges.pkl', 'w'))


def get_features_parallel(samples, label, save_file):

    num_processes = 32

    results = []
    start = time.time()
    for i in range(0, len(samples), num_processes):
        batch = samples[i:i + num_processes]
        jobs = []
        output = multiprocessing.Queue()
        for entry in batch:
            repo_id = entry[0]
            user_login_id = entry[1]
            follower_login_id = entry[2]
            process = multiprocessing.Process(target=get_feature, args=(user_login_id, follower_login_id, repo_id, label, output))
            jobs.append(process)
        for j in jobs:
            j.start()
        for j in jobs:
            j.join()
        outputs = [output.get() for j in jobs]
        results.extend(outputs)
        print(len(results))
    pickle.dump(results, open(save_file + '.pkl', 'wb'))
    print("time : ", time.time() - start)


def get_features_topological_parallel(samples, label, save_file):
    node_mapping, nkt_graph = build_nkt_graph()

    num_processes = 32

    results = []
    start = time.time()
    for i in range(0, len(samples), num_processes):
        batch = samples[i:i + num_processes]
        jobs = []
        output = multiprocessing.Queue()
        for entry in batch:
            repo_id = entry[0]
            user_login_id = entry[1]
            follower_login_id = entry[2]
            process = multiprocessing.Process(target=get_feature_topological, args=(node_mapping, nkt_graph, user_login_id, follower_login_id, repo_id, label, output))
            jobs.append(process)
        for j in jobs:
            j.start()
        for j in jobs:
            j.join()
        outputs = [output.get() for j in jobs]
        results.extend(outputs)
        print(len(results))
    pickle.dump(results, open(save_file + '.pkl', 'wb'))
    print("time : ", time.time() - start)


def get_features_joint_parallel(samples, label, save_file):
    node_mapping, nkt_graph = build_nkt_graph()

    num_processes = 32

    results = []
    start = time.time()
    for i in range(0, len(samples), num_processes):
        batch = samples[i:i + num_processes]
        jobs = []
        output = multiprocessing.Queue()
        for entry in batch:
            repo_id = entry[0]
            user_login_id = entry[1]
            follower_login_id = entry[2]
            process = multiprocessing.Process(target=get_feature_joint, args=(node_mapping, nkt_graph, user_login_id, follower_login_id, repo_id, label, output))
            jobs.append(process)
        for j in jobs:
            j.start()
        for j in jobs:
            j.join()
        outputs = [output.get() for j in jobs]
        results.extend(outputs)
        print(len(results))
    pickle.dump(results, open(save_file + '.pkl', 'wb'))
    print("time : ", time.time() - start)


def convert_timestamp(timestamp):
    if timestamp is None:
        return -1
    date = datetime.datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%SZ')
    epoch = datetime.datetime.fromtimestamp(0)
    return (date - epoch).total_seconds()


def get_user_features(user_login_id):
    user = get_user_profile(user_login_id)
    followers = user['followers'] if user['followers'] is not None else 0
    following = user['following'] if user['following'] is not None else 0
    repo_count = user['public_repos'] if user['public_repos'] is not None else 0
    created_at = convert_timestamp(user['created_at'])
    is_admin = 1 if user['site_admin'] == True else 0
    return [followers, following, repo_count, created_at, is_admin]


def get_repo_features(repo_id):
    repo = get_repo_profile(repo_id)
    created_at = convert_timestamp(repo['created_at'])
    fork_count = repo['forks_count'] if repo['forks_count'] is not None else 0
    total_issues = repo['issue_count']['total_issues_count'] if (repo['issue_count'] is not None and repo['issue_count']['total_issues_count'] is not None) else 0
    open_issues = repo['issue_count']['open_issues_count'] if (repo['issue_count'] is not None and repo['issue_count']['open_issues_count'] is not None) else 0
    watchers = repo['watchers_count'] if repo['watchers_count'] is not None else 0
    return [fork_count, total_issues, open_issues, watchers, created_at]


def get_feature(user_login_id, follower_login_id, repo_id, label, output):
    try:
        features_user = get_user_features(user_login_id)
        features_follower = get_user_features(follower_login_id)
        features_repo = get_repo_features(repo_id)
        is_user_watching_repo = 1 if is_watching(user_login_id, repo_id) else 0
        is_follower_watching_repo = 1 if is_watching(follower_login_id, repo_id) else 0
        feature = np.hstack((features_user, features_repo, features_follower, is_user_watching_repo, is_follower_watching_repo)).ravel()
    except Exception:
        print("Got an Exception")
        output.put((None, None))
        return
    output.put((feature, label))


def get_feature_topological(node_mapping, nkt_graph, user_login_id, follower_login_id, repo_id, label, output):
    try:
        features_edge1 = get_topological_features(nkt_graph, node_mapping, user_login_id, follower_login_id)
        features_edge2 = get_topological_features(nkt_graph, node_mapping, user_login_id, repo_id)
        features_edge3 = get_topological_features(nkt_graph, node_mapping, repo_id, follower_login_id)
        feature = np.hstack((features_edge1, features_edge2, features_edge3)).ravel()
    except Exception:
        print("Got an Exception")
        output.put((None, None))
        return
    output.put((feature, label))


def get_feature_joint(node_mapping, nkt_graph, user_login_id, follower_login_id, repo_id, label, output):
    try:
        features_user = get_user_features(user_login_id)
        features_follower = get_user_features(follower_login_id)
        features_repo = get_repo_features(repo_id)
        is_user_watching_repo = 1 if is_watching(user_login_id, repo_id) else 0
        is_follower_watching_repo = 1 if is_watching(follower_login_id, repo_id) else 0
        features_edge1 = get_topological_features(nkt_graph, node_mapping, user_login_id, follower_login_id)
        features_edge2 = get_topological_features(nkt_graph, node_mapping, user_login_id, repo_id)
        features_edge3 = get_topological_features(nkt_graph, node_mapping, repo_id, follower_login_id)
        feature = np.hstack((features_user, features_repo, features_follower, is_user_watching_repo, is_follower_watching_repo, features_edge1, features_edge2, features_edge3)).ravel()
    except Exception:
        print("Got an Exception")
        output.put((None, None))
        return
    output.put((feature, label))


def get_topological_features(nkt_graph, node_mapping, node0, node1):
    jaccard = nkt.linkprediction.JaccardIndex(nkt_graph).run(node_mapping[node0], node_mapping[node1])
    adamic_adar = nkt.linkprediction.AdamicAdarIndex(nkt_graph).run(node_mapping[node0], node_mapping[node1])
    katz = nkt.linkprediction.KatzIndex(nkt_graph).run(node_mapping[node0], node_mapping[node1])
    common_neighbor = nkt.linkprediction.CommonNeighborsIndex(nkt_graph).run(node_mapping[node0], node_mapping[node1])
    preferential = nkt.linkprediction.PreferentialAttachmentIndex(nkt_graph).run(node_mapping[node0], node_mapping[node1])
    total_neighbors = nkt.linkprediction.TotalNeighborsIndex(nkt_graph).run(node_mapping[node0], node_mapping[node1])
    return [jaccard, adamic_adar, katz, common_neighbor, preferential, total_neighbors]


def get_training_data():
    g = nx.read_gpickle('/home/social-sim/PycharmProjects/Information_Diffusion/data/graph_2016-08-01_2017-01-01.pkl')
    data = pickle.load(open('/home/social-sim/PycharmProjects/Information_Diffusion/data/node_groups_2016-08-01_2017-01-01.pkl', 'rb'))
    users = data['users']
    repos = data['repos']
    positive_samples = []
    negative_samples = []
    print(len(repos))
    for i, repo_id in enumerate(repos):
        neighbors = g[repo_id].keys()
        sub_graph = g.subgraph(neighbors)
        edges = sub_graph.edges()
        other_users = users.difference(set(sub_graph.nodes()))
        if len(edges) > 0:
            for edge in edges():
                user, follower = edge[0], edge[1]
                other_followers = set(get_follower_list(user)).intersection(other_users)
                positive_samples.append((repo_id, user, follower))
                if len(other_followers) > 0:
                    negative_samples.append((repo_id, user, random.choice(list(other_followers))))
            print(i)
    print(len(positive_samples), len(negative_samples))
    pickle.dump(positive_samples, open('train_data_positive_samples.pkl', 'wb'))
    pickle.dump(negative_samples, open('train_data_negative_samples.pkl', 'wb'))


def get_test_data():
    ground_truth_edges = pickle.load(open('/home/social-sim/PycharmProjects/Information_Diffusion/data/test_edges.pkl', 'rb'))
    g = nx.read_gpickle('/home/social-sim/PycharmProjects/Information_Diffusion/data/graph_2016-08-01_2017-01-01.pkl')
    data = pickle.load(open('/home/social-sim/PycharmProjects/Information_Diffusion/data/node_groups_2016-08-01_2017-01-01.pkl', 'rb'))
    users = data['users']
    repos = data['repos']
    positive_samples = []
    negative_samples = []
    for i, ground_truth_edge in enumerate(ground_truth_edges):
        repo_id = ground_truth_edge[0]
        user_id = ground_truth_edge[1]
        forked_users = set(g[repo_id].keys()).difference(set([user_id]))
        if len(forked_users) > 0:
            sub_graph = g.subgraph(forked_users)
            other_users = users.difference(set(sub_graph.nodes()))
            followers = set(get_follower_list(user_id))
            forked_followers = followers.intersection(sub_graph)
            other_followers = followers.intersection(other_users)
            if len(forked_followers) > 0 and len(other_followers) > 0:
                positive_samples.append((repo_id, user_id, random.choice(list(forked_followers))))
                negative_samples.append((repo_id, user_id, random.choice(list(other_followers))))
                print(i)
    print(len(positive_samples), len(negative_samples))
    pickle.dump(positive_samples, open('test_data_positive_samples.pkl', 'wb'))
    pickle.dump(negative_samples, open('test_data_negative_samples.pkl', 'wb'))


def get_train_features():
    positive_samples = pickle.load(open('/home/social-sim/PycharmProjects/Information_Diffusion/data/train_data_positive_samples.pkl', 'rb'))
    negative_samples = pickle.load(open('/home/social-sim/PycharmProjects/Information_Diffusion/data/train_data_negative_samples.pkl', 'rb'))
    get_features_parallel(positive_samples, 1, 'train_data_positive_features')
    get_features_parallel(negative_samples, 0, 'train_data_negative_features')


def get_test_features():
    positive_samples = pickle.load(open('/home/social-sim/PycharmProjects/Information_Diffusion/data/test_data_positive_samples.pkl', 'rb'))
    negative_samples = pickle.load(open('/home/social-sim/PycharmProjects/Information_Diffusion/data/test_data_negative_samples.pkl', 'rb'))
    print(len(positive_samples), len(negative_samples))
    get_features_parallel(positive_samples, 1, 'test_data_positive_features')
    get_features_parallel(negative_samples, 0, 'test_data_negative_features')


def get_train_features_topological():
    positive_samples = pickle.load(open('/home/social-sim/PycharmProjects/Information_Diffusion/data/train_data_positive_samples.pkl', 'rb'))
    negative_samples = pickle.load(open('/home/social-sim/PycharmProjects/Information_Diffusion/data/train_data_negative_samples.pkl', 'rb'))
    print(len(positive_samples), len(negative_samples))
    get_features_topological_parallel(positive_samples, 1, 'train_data_positive_features_topological')
    get_features_topological_parallel(negative_samples, 0, 'train_data_negative_features_topological')


def get_test_features_topological():
    positive_samples = pickle.load(open('/home/social-sim/PycharmProjects/Information_Diffusion/data/test_data_positive_samples.pkl', 'rb'))
    negative_samples = pickle.load(open('/home/social-sim/PycharmProjects/Information_Diffusion/data/test_data_negative_samples.pkl', 'rb'))
    print(len(positive_samples), len(negative_samples))
    get_features_topological_parallel(positive_samples, 1, 'test_data_positive_features_topological')
    get_features_topological_parallel(negative_samples, 0, 'test_data_negative_features_topological')


def get_train_features_joint():
    positive_samples = pickle.load(open('/home/social-sim/PycharmProjects/Information_Diffusion/data/train_data_positive_samples.pkl', 'rb'))
    negative_samples = pickle.load(open('/home/social-sim/PycharmProjects/Information_Diffusion/data/train_data_negative_samples.pkl', 'rb'))
    print(len(positive_samples), len(negative_samples))
    get_features_joint_parallel(positive_samples, 1, 'train_data_positive_features_joint')
    get_features_joint_parallel(negative_samples, 0, 'train_data_negative_features_joint')


def get_test_features_joint():
    positive_samples = pickle.load(open('/home/social-sim/PycharmProjects/Information_Diffusion/data/test_data_positive_samples.pkl', 'rb'))
    negative_samples = pickle.load(open('/home/social-sim/PycharmProjects/Information_Diffusion/data/test_data_negative_samples.pkl', 'rb'))
    print(len(positive_samples), len(negative_samples))
    get_features_joint_parallel(positive_samples, 1, 'test_data_positive_features_joint')
    get_features_joint_parallel(negative_samples, 0, 'test_data_negative_features_joint')


if __name__ =='__main__':
    get_train_features_joint()
    get_test_features_joint()
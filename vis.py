import pickle

def load_results(path):
    with open(path, 'rb') as fp:
        results = pickle.load(fp)
    return results

def main():
    # hyper-parameter
    t = 199

    # load data
    results = load_results('data/sanity-check/0.pkl')
    result = results[t]

    print(result[0].x)

if __name__ == '__main__':
    main()

import random

error_subs = [88, 89, 92, 100]
num_subs = 109

def main(test_ratio = 0.2):
    all_sub_index = set([i for i in range(1, num_subs+1)]).difference(set(error_subs))
    all_subs = ['S{:03}'.format(i) for i in all_sub_index]
    
    num_train = int((num_subs - len(error_subs)) * (1 - test_ratio))
    
    train_subs = random.sample(all_subs, num_train)
    test_subs = test_subs = set(all_subs).difference(set(train_subs))
    
    with open('../config/train_subs.txt', 'w') as f:
        for i in train_subs:
            f.write(i + '\n')
    
    with open('../config/test_subs.txt', 'w') as f:
        for i in test_subs:
            f.write(i + '\n')
            
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--ratio', dest="test_ratio", type = str, help="Test Ratio")

    args = parser.parse_args()
    
    if args.test_ratio
        test_ratio = float(test_ratio)
    else:
        test_ratio = 0.2
        
    main(test_ratio)
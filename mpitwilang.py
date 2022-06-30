from collections import Counter
from mpi4py import MPI
import json
import numpy as np
import os
import sys
import time

np.set_printoptions(linewidth=np.nan)
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


def get_location(grid_path='sydGrid.json'):
    with open(grid_path) as file:
        data = json.load(file)
    coordinates = []
    x = []
    y = []
    letter = ['D', 'C', 'B', 'A']
    corresponding = dict()
    for each in data["features"]:
        text = each['geometry']['coordinates']
        coordinates.append(text[0])
    # print(coordinates)
    for index in range(len(coordinates)):
        # print("---")
        # print(coordinate)
        # print(coordinates[index])
        for each in coordinates[index]:
            x.append(each[0])
            y.append(each[1])
    x = list(set(x))
    y = list(set(y))
    x.sort()
    y.sort()
    # print(x)
    # print(y)
    # corresponding['D1'] = ([x[0],x[1]],[y[0][1]])
    # corresponding['C1'] = ([x[0],x[1]],[y[1][2]])
    # letter = ['D', 'C', 'B', 'A']
    for index in range(len(x)):  # 5
        y_axis = 0
        for each in letter:
            area_code = each + str(index + 1)
            # print(x[index])
            if index < len(x) - 1:
                corresponding[area_code] = ([x[index], x[index + 1]], [y[y_axis], y[y_axis + 1]])
            y_axis += 1
    return corresponding


def deal_twitter(each_tweets, corresponding):
    coordinates = each_tweets['doc']['coordinates']
    area = ""
    if not isinstance(coordinates, type(None)):
        x, y = coordinates['coordinates'][0], coordinates['coordinates'][1]
        # print(x,y)
        language = each_tweets['doc']['lang']
        if language == 'zh-cn' or language == 'zh-tw':
            language = 'zh'
        if language == 'und':
            return None
        for k in corresponding.keys():
            left, right = corresponding[k][0][0], corresponding[k][0][1]
            down, up = corresponding[k][1][0], corresponding[k][1][1]
            if k == 'D1':
                if left <= x <= right and down <= y <= up:
                    area = k
                    break
            elif k in ['A1', 'B1', 'C1']:
                if left <= x <= right and down < y <= up:
                    area = k
                    break
            elif k in ['D2', 'D3', 'D4']:
                if left < x <= right and down <= y <= up:
                    area = k
                    break
            else:
                if left < x <= right and down < y <= up:
                    area = k
                    break
        return area, language


def load_twitter(file_name, corresponding):
    tweets_info = {'A1': [0, set()], 'A2': [0, set()], 'A3': [0, set()], 'A4': [0, set()],
                   'B1': [0, set()], 'B2': [0, set()], 'B3': [0, set()], 'B4': [0, set()],
                   'C1': [0, set()], 'C2': [0, set()], 'C3': [0, set()], 'C4': [0, set()],
                   'D1': [0, set()], 'D2': [0, set()], 'D3': [0, set()], 'D4': [0, set()]}
    language_code = dict()
    for key in tweets_info.keys():
        language_code[key] = []
    # print(language_code)
    number_lines = 0
    # corresponding = get_location()
    # print(corresponding)
    with open(file_name, encoding="utf-8") as file:
        # total_num = file.readlines()
        try:
            file_size = os.path.getsize(file_name)
            chunk_size = file_size // size
            file.seek(rank * chunk_size)
            file.readline()
            while True:
                line = file.readline()  # read line by line in case of memory overflow
                number_lines += 1
                if line:
                    line = line.strip()
                    if len(line) <= 3:
                        continue
                    if line[-3] == ']':
                        target = json.loads(line[:-3])
                    elif line.endswith(','):
                        target = json.loads(line[:-1])
                    elif line.endswith(" "):
                        target = json.loads(line[:-1])
                    elif line.endswith('}}'):
                        target = json.loads(line)
                    else:
                        target = json.loads(line[:-2])

                    # result = ('A1', 'en')
                    result = deal_twitter(target, corresponding)
                    if not isinstance(result, type(None)):
                        area, lan = result
                        if area in tweets_info:
                            tweets_info[area][0] += 1  # may cause out of memory?
                            tweets_info[area][1].add(lan)
                            language_code[area].append(lan)
                    if file.tell() > (rank + 1) * chunk_size:
                        break
                else:
                    break
            print()
        except Exception as e:
            print("Load json function error")
            print("The error in number {} lines".format(number_lines))
            print("The error is ", e)
            file.close()

    # for k, v in tweets_info.items():
    #    print(k, v[0], len(v[1]))  # area, number of tweets, number of language
    language_sum = dict()
    # print(language_code)
    for k, v in language_code.items():
        new_v = dict(Counter(v))
        language_sum[k] = new_v
    return tweets_info, language_sum


def run(data_filename='tinyTwitter.json'):
    lang_dict = {'am': 'Amharic', 'de': 'German', 'ml': 'Malayalam', 'sk': 'Slovak', 'ar': 'Arabic', 'el': 'Greek',
                'dv': 'Maldivian', 'sl': 'Slovenian', 'hy': 'Armenian', 'gu': 'Gujarati', 'mr': 'Marathi',
                'ckb': 'Sorani Kurdish', 'eu': 'Basque', 'ht': 'Haitian Creole', 'ne': 'Nepali', 'es': 'Spanish',
                'bn': 'Bengali', 'iw': 'Hebrew', 'no': 'Norwegian', 'sv': 'Swedish', 'bs': 'Bosnian', 'hi': 'Hindi',
                'or': 'Oriya', 'tl': 'Tagalog', 'bg': 'Bulgarian', 'hi-Latn': 'Latinized Hindi', 'pa': 'Panjabi',
                'ta': 'Tamil', 'my': 'Burmese', 'hu': 'Hungarian', 'ps': 'Pashto', 'te': 'Telugu', 'hr': 'Croatian',
                'is': 'Icelandic', 'fa': 'Persian', 'th': 'Thai', 'ca': 'Catalan', 'in': 'Indonesian', 'pl': 'Polish',
                'bo': 'Tibetan', 'cs': 'Czech', 'it': 'Italian', 'pt': 'Portuguese', 'zh': 'Chinese', 'da': 'Danish',
                'ja': 'Japanese', 'ro': 'Romanian', 'tr': 'Turkish', 'nl': 'Dutch', 'kn': 'Kannada', 'ru': 'Russian',
                'uk': 'Ukrainian', 'en': 'English', 'km': 'Khmer', 'sr': 'Serbian', 'ur': 'Urdu', 'et': 'Estonian',
                'ko': 'Korean', 'ug': 'Uyghur', 'fi': 'Finnish', 'lo': 'Lao', 'sd': 'Sindhi', 'vi': 'Vietnamese',
                'fr': 'French', 'lv': 'Latvian', 'si': 'Sinhala', 'cy': 'Welsh', 'ka': 'Georgian', 'lt': 'Lithuanian'}

    area_location = get_location('sydGrid.json')
    tweet_info, lan = load_twitter(data_filename, area_location)
    number_tweets = []
    languages = []
    for k, v in tweet_info.items():
        number_tweets.append(v[0])
        languages.append(','.join(list(v[1])))
    lan_list = []
    for k, v in lan.items():
        dict_string = json.dumps(v)
        lan_list.append(dict_string)

    # print('before gathering', number_tweets)
    # print('before gathering', languages)
    # print('before gathering', lan_list)

    number_tweets = np.array(number_tweets, dtype='i')
    languages = np.array(languages, dtype='S120').tobytes()
    lan_list = np.array(lan_list, dtype='S400').tobytes()

    tweets_buff = None
    languages_buff = None
    lanlist_buff = None
    if rank == 0:
        tweets_buff = np.empty(size * 16, dtype='i')
        languages_buff = bytearray(size * 1920)
        lanlist_buff = bytearray(size * 6400)

    comm.Gather(number_tweets, tweets_buff, root=0)
    comm.Gather(languages, languages_buff, root=0)
    comm.Gather(lan_list, lanlist_buff, root=0)

    if not rank == 0:
        return

    # Only rank 0 runs the following code
    tweets = np.reshape(tweets_buff, (-1, 16))
    tweets = np.sum(tweets, axis=0)

    gathered_languages = np.frombuffer(languages_buff, dtype='S120', count=16 * size)
    gathered_languages = np.reshape(gathered_languages, (-1, 16))
    gathered_languages = list(zip(*gathered_languages))
    for i, k in enumerate(tweet_info.keys()):
        lang_set = set()
        for lang_str in gathered_languages[i]:
            if lang_str == b'':
                continue
            lang_list = lang_str.split(b',')
            for lang_ in lang_list:
                lang_set.add(lang_)
        tweet_info[k] = [tweets[i], lang_set]

    gathered_lanlist = np.frombuffer(lanlist_buff, dtype='S400', count=16 * size)
    gathered_lanlist = np.reshape(gathered_lanlist, (-1, 16))
    gathered_lanlist = list(zip(*gathered_lanlist))
    for i, k in enumerate(lan.keys()):
        lan_dict = {}
        for lan_dict_b in gathered_lanlist[i]:
            lan_dict_ = json.loads(lan_dict_b.decode())
            for lan_k, lan_v in lan_dict_.items():
                if lan_dict.get(lan_k):
                    lan_dict[lan_k] += lan_v
                else:
                    lan_dict[lan_k] = lan_v
        lan[k] = lan_dict

    # print('after gathering\n', tweets)
    # print('after gathering\n', gathered_languages)
    # print('after gathering\n', gathered_lanlist)
    # print('after gathering\n', tweet_info)
    # print('after gathering\n', lan)

    print('Cell\tTotal Tweets\t#Number of Languages Used\t\t#Top 10 Languages & #Tweets)')
    for k, v in tweet_info.items():
        print("%-10s\t%d\t\t\t%d" % (k, v[0], len(v[1])), end="")

        temp = "("
        summary = lan[k]
        summary = {k: v for k, v in sorted(summary.items(), key=lambda item: item[1], reverse=True)}
        if len(summary) > 10:
            count = 0
            for language, frequency in summary.items():
                print_lang = lang_dict[language]
                if count == 0:
                    temp += print_lang + "-" + str(frequency)
                elif count < 10:
                    temp += ', ' + print_lang + "-" + str(frequency)
                else:
                    break
                count += 1
        else:
            for idx, (language, frequency) in enumerate(summary.items()):
                print_lang = lang_dict[language]
                if idx == 0:
                    temp += print_lang + "-" + str(frequency)
                else:
                    temp += ', ' + print_lang + "-" + str(frequency)
                # print(temp)
        temp += ")"
        print("\t\t\t%s" % temp)


if __name__ == '__main__':
    # start = time.time()
    args = sys.argv
    filename = os.path.basename(__file__)
    twitter_filename = None
    for i, arg in enumerate(args):
        if os.path.basename(arg) == filename:
            if not i == len(args) - 1:
                twitter_filename = args[i + 1]
                if not os.path.exists(twitter_filename):
                    twitter_filename = None
                    print('Warning: Invalid data path. Using default tweet data.')
                    break
            else:
                print('Warning: Using default tweet data.')
            break
    if twitter_filename:
        run(twitter_filename)
    else:
        run()
    # end = time.time()
    # if rank == 0:
    #     print('Time consuming: {:.3f}s'.format(end - start))

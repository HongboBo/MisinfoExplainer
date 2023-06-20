import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib.ticker as ticker
import json


plt.style.use('ggplot')

dict = {'tweet_text':0.0812,  'tweet_lang':0.0535,  'claim_embedding':0.0718,  'claim_reviewer':0.0531,  'image':0.0683,
        'hashtag':0.0338,  'user':0.0602,  'reply_text':0.0812,  'reply_lang':0.0477,  'tweet_discusses_claim':0.0381,
        'tweet_has_hashtag_hashtag':0.0370,  'user_has_hashtag_hashtag':0.0376,  'tweet_has_image_image':0.0335,
        'user_posted_reply':0.0389,  'reply_quote_of_tweet':0.0390,  'reply_reply_to_tweet':0.0390,  'tweet_mentions_user':0.0358,
        'user_posted_tweet':0.0378,  'user_retweeted_tweet':0.0387,  'user_follows_user':0.0386,  'user_mentions_user':0.0353}

dict2 =  {'tweet_text': 0.0706, 'tweet_lang': 0.0410, 'claim_embedding': 0.0497, 'claim_reviewer': 0.0390, 'image': 0.0517,
          'hashtag': 0.0379, 'user': 0.0382, 'reply_text': 0.0528, 'reply_lang': 0.0396, 'tweet_discusses_claim': 0.0425,
          'tweet_has_hashtag_hashtag': 0.0559, 'user_has_hashtag_hashtag': 0.0611, 'tweet_has_image_image': 0.0368,
          'user_posted_reply': 0.0368, 'reply_quote_of_tweet': 0.0368, 'reply_reply_to_tweet': 0.0368, 'tweet_mentions_user': 0.0550,
          'user_posted_tweet': 0.0368, 'user_retweeted_tweet': 0.0640, 'user_follows_user': 0.0676, 'user_mentions_user': 0.0496}


print(sorted(dict2.items(), key = lambda kv:(kv[1], kv[0])))
exit()




term = []
for i in range(21):
        if i < 9:
                term.append('n'+str(i+1))
        else:
                term.append('e' + str(i-8))
d = np.array(list(dict.keys()))

# tmp = []
# for i in range(21):
#         if '_' in d[i]:
#                 d[i] = d[i].replace('_', '\_')
#         tmp.append(term[i] + ' & ' + d[i] +' ')
# tmp.append('  &  ')
# for i in range(17):
#         print(tmp[2*i] + ' & '+ tmp[2*i+1] + '\\\\')


im = np.array(list(dict.values()))
im2 = np.array(list(dict2.values()))
# im2 = np.pad(im2, (0, 24), 'constant', constant_values=(0, 0))

bar_width = 0.4
bar_x =np.arange(len(term))


fig = plt.figure(figsize=(12,5))
plt.rcParams.update({'font.size': 15})

ax = fig.add_subplot(111)
bar1 = ax.bar(x=bar_x - bar_width/2,
              height=im, width=bar_width)
bar2 = ax.bar(x=bar_x + bar_width/2,
              height=im2, width=bar_width,
        )

order = ['#1', '#2', '#3', '#4', '#5']
highlight_idx = [0, 7, 2, 4, 6]
highlight_idx2 = [0, 19, 18, 11, 10]
for i in range(5):
    ax.text(bar_x[highlight_idx[i]] - bar_width/2, im[highlight_idx[i]]+0.001, order[i], ha='center', color='red', fontsize=11)
    ax.text(bar_x[highlight_idx2[i]] + bar_width/2, im2[highlight_idx2[i]]+0.001, order[i], ha='center', color='blue', fontsize=11)


ax.set_xlabel('Inputs')
plt.ylabel('Im')

ax.set_xticks(range(21))
ax.set_xticklabels(term)
ax.legend((bar1, bar2), ('Perturbation', 'Gradient'), fontsize=11)


plt.savefig("test.pdf", bbox_inches='tight')
plt.show()
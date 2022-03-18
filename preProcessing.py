# Loading library
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import numpy as np
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

# Read Dataset
youtubeData = pd.read_csv("VideoLikesDataset.csv")
# youtube.head()
# print(youtube.head())

# print(youtube.shape)

# print(youtubeData.isnull().values.any())

youtubeData = youtubeData.dropna(how='any', axis=0)

# print(youtube.describe())


youtubeData.drop(['video_id'], axis=1, inplace=True)
# youtubeData.apply(lambda x: len(x.unique()))

# for x in (['comments_disabled', 'ratings_disabled', 'video_error_or_removed', 'category_id']):
#     count = youtube[x].value_counts()
#     print(count)
#     plt.figure(figsize=(7, 7))
#     sns.barplot(count.index, count.values, alpha=0.8)
#     plt.title('{} vs No of video'.format(x))
#     plt.ylabel('No of video')
#     plt.xlabel('{}'.format(x))
#     plt.show()

# No of tags
tags = [x.count("|") + 1 for x in youtubeData["tags"]]
youtubeData["tags_count"] = tags

# length of description
desc_len = [len(x) for x in youtubeData["video_description"]]
youtubeData["desc_len"] = desc_len

# length of title
title_len = [len(x) for x in youtubeData["title"]]
youtubeData["title_len"] = title_len

publish_time = pd.to_datetime(youtubeData['publish_time'], format='%Y-%m-%dT%H:%M:%S.%fZ')
youtubeData['publish_time'] = publish_time.dt.time
youtubeData['publish_date'] = publish_time.dt.date

# day at which video is published
youtubeData['publish_weekday'] = publish_time.dt.dayofweek
# monday is the start of the week

# ratio of view/comment_count  upto 3 decimal
youtubeData["View_per_comment"] = round(youtubeData["views"] / youtubeData["comment_count"], 3)

# print(max(youtubeData["Ratio_views_comment_count"]))

# removing the infinite values
youtubeData = youtubeData.replace([np.inf, -np.inf], np.nan)
youtubeData = youtubeData.dropna(how='any', axis=0)

data = youtubeData

corr = data.corr()
plt.figure(figsize=(16, 16))
ax = sns.heatmap(corr, vmin=-0.5, vmax=0.5, center=0, square=True, cmap=sns.diverging_palette(10, 220, n=200))
# plt.show()

# views comment_count , [likes , comment_count & views ]

youtubeData.drop(
    ['trending_date', 'publish_date', 'publish_time', 'tags', 'title', 'video_description', 'channel_title'],
    axis=1,
    inplace=True)

# youtubeData = youtubeData.sample(frac=1).reset_index(drop=True)

likes = youtubeData['likes']
youtube_like = youtubeData.drop(['likes'], axis=1, inplace=False)

# likes = likes.sample(frac=1).reset_index(drop=True)
# youtube_like = youtube_like.sample(frac=1).reset_index(drop=True)

# youtube_like = shuffle(youtube_like)
# likes = shuffle(likes)

x_train, x_test, y_train, y_test = train_test_split(youtube_like, likes, test_size=0.2, shuffle=False)


def get_data_set():
    return x_train, x_test, y_train, y_test
# print(train.shape, test.shape, y_train.shape, y_test.shape)

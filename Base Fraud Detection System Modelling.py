import matplotlib.pyplot as plt
from PIL._imaging import display

from EDA import *;
from SharedFunctions import *;


# 4.1. Defining the training and test sets¶

# Compute the number of transactions per day, fraudulent transactions per day and fraudulent cards per day

def get_tx_stats(transactions_df, start_date_df="2018-04-01"):
    # Number of transactions per day
    nb_tx_per_day = transactions_df.groupby(['TX_TIME_DAYS'])['CUSTOMER_ID'].count()
    # Number of fraudulent transactions per day
    nb_fraudulent_transactions_per_day = transactions_df.groupby(['TX_TIME_DAYS'])['TX_FRAUD'].sum()
    # Number of compromised cards per day
    nb_compromised_cards_per_day = transactions_df[transactions_df['TX_FRAUD'] == 1].groupby(
        ['TX_TIME_DAYS']).CUSTOMER_ID.nunique()

    tx_stats = pd.DataFrame({"nb_tx_per_day": nb_tx_per_day,
                             "nb_fraudulent_transactions_per_day": nb_fraudulent_transactions_per_day,
                             "nb_compromised_cards_per_day": nb_compromised_cards_per_day})

    tx_stats = tx_stats.reset_index()

    start_date = datetime.datetime.strptime(start_date_df, "%Y-%m-%d")
    tx_date = start_date + tx_stats['TX_TIME_DAYS'].apply(datetime.timedelta)

    tx_stats['tx_date'] = tx_date

    return tx_stats


tx_stats = get_tx_stats(transactions_df, start_date_df="2018-04-01")


# Plot the number of transactions per day, fraudulent transactions per day and fraudulent cards per day

def get_template_tx_stats(ax, fs,
                          start_date_training,
                          title='',
                          delta_train=7,
                          delta_delay=7,
                          delta_test=7,
                          ylim=300):
    ax.set_title(title, fontsize=fs * 1.5)
    ax.set_ylim([0, ylim])

    ax.set_xlabel('Date', fontsize=fs)
    ax.set_ylabel('Number', fontsize=fs)

    plt.yticks(fontsize=fs * 0.7)
    plt.xticks(fontsize=fs * 0.7)

    ax.axvline(start_date_training + datetime.timedelta(days=delta_train), 0, ylim, color="black")
    ax.axvline(start_date_test, 0, ylim, color="black")

    ax.text(start_date_training + datetime.timedelta(days=2), ylim - 20, 'Training period', fontsize=fs)
    ax.text(start_date_training + datetime.timedelta(days=delta_train + 2), ylim - 20, 'Delay period', fontsize=fs)
    ax.text(start_date_training + datetime.timedelta(days=delta_train + delta_delay + 2), ylim - 20, 'Test period',
            fontsize=fs)


cmap = plt.get_cmap('jet')
colors = {'nb_tx_per_day': cmap(0),
          'nb_fraudulent_transactions_per_day': cmap(200),
          'nb_compromised_cards_per_day': cmap(250)}

fraud_and_transactions_stats_fig, ax = plt.subplots(1, 1, figsize=(15, 8))

# Training period
start_date_training = datetime.datetime.strptime("2018-07-25", "%Y-%m-%d")
delta_train = delta_delay = delta_test = 7

end_date_training = start_date_training + datetime.timedelta(days=delta_train - 1)

# Test period
start_date_test = start_date_training + datetime.timedelta(days=delta_train + delta_delay)
end_date_test = start_date_training + datetime.timedelta(days=delta_train + delta_delay + delta_test - 1)

get_template_tx_stats(ax, fs=20,
                      start_date_training=start_date_training,
                      title='Total transactions, and number of fraudulent transactions \n and number of compromised cards per day',
                      delta_train=delta_train,
                      delta_delay=delta_delay,
                      delta_test=delta_test
                      )

ax.plot(tx_stats['tx_date'], tx_stats['nb_tx_per_day'] / 50, 'b', color=colors['nb_tx_per_day'],
        label='# transactions per day (/50)')
ax.plot(tx_stats['tx_date'], tx_stats['nb_fraudulent_transactions_per_day'], 'b',
        color=colors['nb_fraudulent_transactions_per_day'], label='# fraudulent txs per day')
ax.plot(tx_stats['tx_date'], tx_stats['nb_compromised_cards_per_day'], 'b',
        color=colors['nb_compromised_cards_per_day'], label='# compromised cards per day')

ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=20)


# plt.show(fraud_and_transactions_stats_fig)


def get_train_test_set(transactions_df,
                       start_date_training,
                       delta_train=7, delta_delay=7, delta_test=7):
    # Get the training set data
    train_df = transactions_df[(transactions_df.TX_DATETIME >= start_date_training) &
                               (transactions_df.TX_DATETIME < start_date_training + datetime.timedelta(
                                   days=delta_train))]

    # Get the test set data
    test_df = []

    # Note: Cards known to be compromised after the delay period are removed from the test set
    # That is, for each test day, all frauds known at (test_day-delay_period) are removed

    # First, get known defrauded customers from the training set
    known_defrauded_customers = set(train_df[train_df.TX_FRAUD == 1].CUSTOMER_ID)

    # Get the relative starting day of training set (easier than TX_DATETIME to collect test data)
    start_tx_time_days_training = train_df.TX_TIME_DAYS.min()

    # Then, for each day of the test set
    for day in range(delta_test):
        # Get test data for that day
        test_df_day = transactions_df[transactions_df.TX_TIME_DAYS == start_tx_time_days_training +
                                      delta_train + delta_delay +
                                      day]

        # Compromised cards from that test day, minus the delay period, are added to the pool of known defrauded customers
        test_df_day_delay_period = transactions_df[transactions_df.TX_TIME_DAYS == start_tx_time_days_training +
                                                   delta_train +
                                                   day - 1]

        new_defrauded_customers = set(test_df_day_delay_period[test_df_day_delay_period.TX_FRAUD == 1].CUSTOMER_ID)
        known_defrauded_customers = known_defrauded_customers.union(new_defrauded_customers)

        test_df_day = test_df_day[~test_df_day.CUSTOMER_ID.isin(known_defrauded_customers)]

        test_df.append(test_df_day)

    test_df = pd.concat(test_df)

    # Sort data sets by ascending order of transaction ID
    train_df = train_df.sort_values('TRANSACTION_ID')
    test_df = test_df.sort_values('TRANSACTION_ID')

    return (train_df, test_df)


(train_df, test_df) = get_train_test_set(transactions_df, start_date_training,
                                         delta_train=7, delta_delay=7, delta_test=7)

print("Train Data Shape :- ", train_df.shape)

print("Number of fraudulent translation in train data :- ", train_df[train_df.TX_FRAUD == 1].shape)

print("Test data shape :- ", test_df.shape)

## 4.2. Model training : Decision tree¶


output_feature = "TX_FRAUD"

input_features = ['TX_AMOUNT', 'TX_DURING_WEEKEND', 'TX_DURING_NIGHT', 'CUSTOMER_ID_NB_TX_1DAY_WINDOW',
                  'CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW', 'CUSTOMER_ID_NB_TX_7DAY_WINDOW',
                  'CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW', 'CUSTOMER_ID_NB_TX_30DAY_WINDOW',
                  'CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW', 'TERMINAL_ID_NB_TX_1DAY_WINDOW',
                  'TERMINAL_ID_RISK_1DAY_WINDOW', 'TERMINAL_ID_NB_TX_7DAY_WINDOW',
                  'TERMINAL_ID_RISK_7DAY_WINDOW', 'TERMINAL_ID_NB_TX_30DAY_WINDOW',
                  'TERMINAL_ID_RISK_30DAY_WINDOW']


def fit_model_and_get_predictions(classifier, train_df, test_df,
                                  input_features, output_feature="TX_FRAUD", scale=True):
    # By default, scales input data
    if scale:
        (train_df, test_df) = scaleData(train_df, test_df, input_features)

    # We first train the classifier using the `fit` method, and pass as arguments the input and output features
    start_time = time.time()
    classifier.fit(train_df[input_features], train_df[output_feature])
    training_execution_time = time.time() - start_time

    # We then get the predictions on the training and test data using the `predict_proba` method
    # The predictions are returned as a numpy array, that provides the probability of fraud for each transaction
    start_time = time.time()
    predictions_test = classifier.predict_proba(test_df[input_features])[:, 1]
    prediction_execution_time = time.time() - start_time

    predictions_train = classifier.predict_proba(train_df[input_features])[:, 1]

    # The result is returned as a dictionary containing the fitted models,
    # and the predictions on the training and test sets
    model_and_predictions_dictionary = {'classifier': classifier,
                                        'predictions_test': predictions_test,
                                        'predictions_train': predictions_train,
                                        'training_execution_time': training_execution_time,
                                        'prediction_execution_time': prediction_execution_time
                                        }

    return model_and_predictions_dictionary


# We first create a decision tree object. We will limit its depth to 2 for interpretability,
# and set the random state to zero for reproducibility
classifier = sklearn.tree.DecisionTreeClassifier(max_depth=2, random_state=0)

model_and_predictions_dictionary = fit_model_and_get_predictions(classifier, train_df, test_df,
                                                                 input_features, output_feature,
                                                                 scale=False)
# print("Let us look at the predictions obtained for the first five transactions of the test set:")

test_df['TX_FRAUD_PREDICTED'] = model_and_predictions_dictionary['predictions_test']


# print(test_df.head())

# display(graphviz.Source(sklearn.tree.export_graphviz(classifier,feature_names=input_features,class_names=True, filled=True)))

# from dtreeplt import dtreeplt
#
# dtree = dtreeplt(model=classifier, feature_names=input_features, target_names=output_feature)
# fig = dtree.view()
# currentFigure = plt.gcf()
# currentFigure.set_size_inches(50, 20)

# plt.show(tree.plot_tree(classifier))

# from IPython.display import display
#
# fraud_and_transactions_stats_fig2, ax2 = display(graphviz.Source(tree.export_graphviz(classifier)))
# plt.show(fraud_and_transactions_stats_fig2)

## 4.3. performance assessment¶

def card_precision_top_k_day(df_day, top_k):
    # This takes the max of the predictions AND the max of label TX_FRAUD for each CUSTOMER_ID,
    # and sorts by decreasing order of fraudulent prediction
    df_day = df_day.groupby('CUSTOMER_ID').max().sort_values(by="predictions", ascending=False).reset_index(drop=False)

    # Get the top k most suspicious cards
    df_day_top_k = df_day.head(top_k)
    list_detected_compromised_cards = list(df_day_top_k[df_day_top_k.TX_FRAUD == 1].CUSTOMER_ID)

    # Compute precision top k
    card_precision_top_k = len(list_detected_compromised_cards) / top_k

    return list_detected_compromised_cards, card_precision_top_k


def card_precision_top_k(predictions_df, top_k, remove_detected_compromised_cards=True):
    # Sort days by increasing order
    list_days = list(predictions_df['TX_TIME_DAYS'].unique())
    list_days.sort()

    # At first, the list of detected compromised cards is empty
    list_detected_compromised_cards = []

    card_precision_top_k_per_day_list = []
    nb_compromised_cards_per_day = []

    # For each day, compute precision top k
    for day in list_days:

        df_day = predictions_df[predictions_df['TX_TIME_DAYS'] == day]
        df_day = df_day[['predictions', 'CUSTOMER_ID', 'TX_FRAUD']]

        # Let us remove detected compromised cards from the set of daily transactions
        df_day = df_day[df_day.CUSTOMER_ID.isin(list_detected_compromised_cards) == False]

        nb_compromised_cards_per_day.append(len(df_day[df_day.TX_FRAUD == 1].CUSTOMER_ID.unique()))

        detected_compromised_cards, card_precision_top_k = card_precision_top_k_day(df_day, top_k)

        card_precision_top_k_per_day_list.append(card_precision_top_k)

        # Let us update the list of detected compromised cards
        if remove_detected_compromised_cards:
            list_detected_compromised_cards.extend(detected_compromised_cards)

    # Compute the mean
    mean_card_precision_top_k = np.array(card_precision_top_k_per_day_list).mean()

    # Returns precision top k per day as a list, and resulting mean
    return nb_compromised_cards_per_day, card_precision_top_k_per_day_list, mean_card_precision_top_k


def performance_assessment(predictions_df, output_feature='TX_FRAUD',
                           prediction_feature='predictions', top_k_list=[100],
                           rounded=True):
    AUC_ROC = metrics.roc_auc_score(predictions_df[output_feature], predictions_df[prediction_feature])
    AP = metrics.average_precision_score(predictions_df[output_feature], predictions_df[prediction_feature])

    performances = pd.DataFrame([[AUC_ROC, AP]],
                                columns=['AUC ROC', 'Average precision'])

    for top_k in top_k_list:
        _, _, mean_card_precision_top_k = card_precision_top_k(predictions_df, top_k)
        performances['Card Precision@' + str(top_k)] = mean_card_precision_top_k

    if rounded:
        performances = performances.round(3)

    return performances


# Let us compute the performance in terms of AUC ROC, Average Precision (AP), and Card Precision top 100 (CP@100) for
# the decision tree.

predictions_df = test_df
predictions_df['predictions'] = model_and_predictions_dictionary['predictions_test']

print("Performance of the Decision Tree", '\n')
print(performance_assessment(predictions_df, top_k_list=[100]), '\n')

# 4.4. Performances using standard prediction models¶

classifiers_dictionary = {'Logistic regression': sklearn.linear_model.LogisticRegression(random_state=0),
                          'Decision tree with depth of two': sklearn.tree.DecisionTreeClassifier(max_depth=2,
                                                                                                 random_state=0),
                          'Decision tree - unlimited depth': sklearn.tree.DecisionTreeClassifier(random_state=0),
                          'Random forest': sklearn.ensemble.RandomForestClassifier(random_state=0, n_jobs=-1),
                          'XGBoost': xgboost.XGBClassifier(random_state=0, n_jobs=-1),
                          }

fitted_models_and_predictions_dictionary = {}

for classifier_name in classifiers_dictionary:
    model_and_predictions = fit_model_and_get_predictions(classifiers_dictionary[classifier_name], train_df, test_df,
                                                          input_features=input_features,
                                                          output_feature=output_feature)
    fitted_models_and_predictions_dictionary[classifier_name] = model_and_predictions


def performance_assessment_model_collection(fitted_models_and_predictions_dictionary,
                                            transactions_df,
                                            type_set='test',
                                            top_k_list=[100]):
    performances = pd.DataFrame()

    for classifier_name, model_and_predictions in fitted_models_and_predictions_dictionary.items():
        predictions_df = transactions_df

        predictions_df['predictions'] = model_and_predictions['predictions_' + type_set]

        performances_model = performance_assessment(predictions_df, output_feature='TX_FRAUD',
                                                    prediction_feature='predictions', top_k_list=top_k_list)
        performances_model.index = [classifier_name]

        performances = performances.append(performances_model)

    return performances


# performances on test set
df_performances = performance_assessment_model_collection(fitted_models_and_predictions_dictionary, test_df,
                                                          type_set='test',
                                                          top_k_list=[100])
print("Performance of Test Data :- ", '\n')
print(df_performances, '\n')

# performances on training set
df_performances = performance_assessment_model_collection(fitted_models_and_predictions_dictionary, train_df,
                                                          type_set='train',
                                                          top_k_list=[100])
print("Performance of Train Data :- ", '\n')
print(df_performances, '\n')


def execution_times_model_collection(fitted_models_and_predictions_dictionary):
    execution_times = pd.DataFrame()

    for classifier_name, model_and_predictions in fitted_models_and_predictions_dictionary.items():
        execution_times_model = pd.DataFrame()
        execution_times_model['Training execution time'] = [model_and_predictions['training_execution_time']]
        execution_times_model['Prediction execution time'] = [model_and_predictions['prediction_execution_time']]
        execution_times_model.index = [classifier_name]

        execution_times = execution_times.append(execution_times_model)

    return execution_times


# Execution times
df_execution_times = execution_times_model_collection(fitted_models_and_predictions_dictionary)
print("Execution time :- ", '\n')
print(df_execution_times)



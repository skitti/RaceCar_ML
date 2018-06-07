import boto3
from boto3.dynamodb.conditions import Key, Attr
import pandas


# # Let's use Amazon S3

#dynamodb = boto3.resource('dynamodb',aws_access_key_id=ACCESS_ID,
#         aws_secret_access_key=ACCESS_KEY, region_name='us-west-2')
#dynamodb = boto3.client('dynamodb', region_name='us-west-2')
# Get the service resource.
dynamodb = boto3.resource('dynamodb')
#print(dynamodb)

# Print out some data about the table.
table = dynamodb.Table('RCcarTable6')
#print(table)

'''
def tablequery1(): #returns some kind of hashfunction
    response = table.query(
        KeyConditionExpression=Key('RCcarNumber').eq(1)
    )
    items = response['Items']
#print(tablequery1) 


def get_table_metadata(table_name): #Table Metadata
    """
    Get some metadata about chosen table.
    """
    table = dynamodb.Table(table_name)

    return{
        'num_items': table.item_count,
        'primary_key_name': table.key_schema[0],
        'status': table.table_status,
        'bytes_size': table.table_size_bytes,
        'global_secondary_indices': table.global_secondary_indexes
          }
#print (get_table_metadata('RCcarTable3'))
'''


def query_table(table_name, filter_key=None, filter_value=None):
    """
    Perform a query operation on the table. 
    Can specify filter_key (col name) and its value to be filtered.
    """
    table = dynamodb.Table(table_name)

    if filter_key and filter_value:
        filtering_exp = Key(filter_key).eq(filter_value)
        response = table.query(KeyConditionExpression=filtering_exp)
    else:
        response = table.query()

    return response
 



def scan_table_allpages(table_name, filter_key=None, filter_value=None):
    """
    Perform a scan operation on table. 
    Can specify filter_key (col name) and its value to be filtered. 
    This gets all pages of results. Returns list of items.
    """
    table = dynamodb.Table(table_name)

    if filter_key and filter_value:
        filtering_exp = Key(filter_key).eq(filter_value)
        response = table.scan(FilterExpression=filtering_exp)
    else:
        response = table.scan()

    items = response['Items']
    while True:
        #print (len(response['Items']))
        if response.get('LastEvaluatedKey'):
            response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
            items += response['Items']
        else:
            break

    return items
    
 
def main_skapadf():
	#querytable = query_table('RCcartable6','RCcarNumber',2)
	#testprint = querytable
	testprint = (scan_table_allpages('RCcarTable6','payload'))
	#print(testprint)
	#print(testprint,"\n",querytable) 

	dataframelist=pandas.DataFrame(testprint)
	#print(dataframelist)
	newdf = dataframelist['payload'].apply(pandas.Series)
	#newdf = newdf.reindex(columns=['payload'][0])
	#print(newdf)
	#newdf = newdf.loc[(newdf['RCcarNumber']==2)]
	#newdf = newdf.loc[(newdf[0]==2)]
	#print(newdf)
	#exit()
	#print(pandas.DataFrame(newdf))
	#print(dataframelist['payload'])

	payloaddataframe = (dataframelist['payload'])
	df = pandas.DataFrame(payloaddataframe)
	return(newdf)



def printallcolumns(df_all):
    #Print out all the columns (features)
    print(df_all.shape[1]) #[0] gives nr of rows [1] gives nr of columns
    for i in range(df_all.shape[1]):
        print(i, df_all.columns[i])
   

#printallcolumns(dataframelist['payload'])

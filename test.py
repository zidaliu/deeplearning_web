
import IDS
# In[7]:
# testfile = 'data/good_fromE2.txt'
# testfile = 'data/goodqueries.txt'
good_testfile = "data/good_fromE2.txt"
bad_testfile = "data/badqueries.txt"
# a = IDS.LG()

a = IDS.SVM()

# preicdtlist = ['www.foo.com/id=1<script>alert(1)</script>','www.foo.com/name=admin\' or 1=1','abc.com/admin.php','"><svg onload=confirm(1)>','test/q=<a href="javascript:confirm(1)>','q=../etc/passwd']
# result =a.predict(preicdtlist)
# print('正常结果 前10条 ' + str(result[0][:10]))



with open(good_testfile, 'r') as f:
    print('预测数据集： '+good_testfile)
    preicdtlist = [i.strip('\n') for i in f.readlines()[:]]
    result = a.predict(preicdtlist)
    print('恶意结果 前10条'+str(result[1][:10]))
    print('正常结果 前10条 ' + str(result[0][:10]))
    pass


with open(bad_testfile, 'r') as f:
    print('预测数据集： '+bad_testfile)
    preicdtlist = [i.strip('\n') for i in f.readlines()[:]]
    result = a.predict(preicdtlist)
    print('恶意结果 前10条'+str(result[1][:10]))
    print('正常结果 前10条 ' + str(result[0][:10]))
    pass
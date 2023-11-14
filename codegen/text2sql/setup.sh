#!/bin/bash

########创建database并导入sql#########

# 创建 RDS MySQL 数据库实例,命名为 mydb,db.t2.micro 类型,用户名 admin,密码为 abc123
aws rds create-db-instance --db-instance-identifier llm2 --engine mysql --db-instance-class db.r6g.large --master-username admin --master-user-password ***** --allocated-storage 100
# 等待数据库实例创建完成并可用
aws rds wait db-instance-available --db-instance-identifier llm2
endpoint=$(aws rds describe-db-instances --db-instance-identifier llm2 --query 'DBInstances[*].Endpoint.Address' --output text)
echo "Endpoint: $endpoint"

# 安装客户端并执行ddl
sudo yum install mysql 
mysql -h $endpoint -u admin -padmin12345678 〈./ddl.sql

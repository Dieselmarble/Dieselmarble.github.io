---
title: Connect to my AWS EC2-Instance
date: 2018-10-31 20:42:41
tags:
categories: Blog
visible:
---

## 链接 AWS EC2 虚拟机

- Use the following command to set the permissions of your private key file so that only you can read it.

> chmod 400 /path/my-key-pair.pem

- Use the ssh command to connect to the instance. You specify the private key (.pem) file and user_name@public_dns_name. For example, if you used Amazon Linux 2 or the Amazon Linux AMI, the user name is ec2-user.

> ssh -i /path/my-key-pair.pem [ec2-user@ec2-198-51-100-1.compute-1.amazonaws.com](mailto:ec2-user@ec2-198-51-100-1.compute-1.amazonaws.com)

- Replace EC2 with Ubuntu if using an ubuntu virtual machine
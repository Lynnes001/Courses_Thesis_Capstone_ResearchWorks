#!/usr/bin/env python
# coding:utf-8
'''
works in python 3.6 environment
Created by Ke Xia, Feb 22 2018
'''
import numpy as np
import csv
import sys

alpha = 0.005
beta = 11.1
N = 1025  # number of t
M = 9  # assume m is 6


def readcsvfile(filename):
    '''
 	read csv file
 	input: filename
 	output: a list stores the close prices in 'filename'
 	'''
    csvfile = open(filename, 'r')
    reader = csv.reader(csvfile)
    result = []
    for item in reader:
        if reader.line_num == 1:
            continue
        result.append(item[4])
    csvfile.close()
    return result


# return new x at time t+1
def bayesiancurvefitting(x, t, newx):
    '''
	x and t are training datasets
	x: time or date
	t: price
	new x: new giving value(time)
	output: parameters of Gaussian distribution of t
	'''
    # compute phi_x and its transpose
    o_xn = np.zeros(M)
    o_x = np.zeros(M)
    for m in range(M):
        o_x[m] = np.power(float(newx), m)
    o_x_T = np.transpose([o_x])

    # compute matrix S
    I = np.identity(M, dtype=float)  # create a identity matrix with size m*m
    right_part = np.zeros((M, M), dtype=float)
    for i in range(N):
        for j in range(M):
            o_xn[j] = np.power(float(x[i]), j)
        o_xn_T = np.transpose([o_xn])
        right_part += np.dot(o_xn_T, [o_x])
    S_inv = np.zeros((M, M), dtype=float)
    S_inv = alpha * I + beta * right_part
    S = np.linalg.inv(S_inv)

    # compute m(x)
    xn_sum = np.zeros((M, 1), dtype=float)
    for k in range(N):
        for l in range(M):
            o_xn[l] = np.power(float(x[k]), l)
        o_xn_T = np.transpose([o_xn])
        xn_sum += float(t[k]) * o_xn_T
    m_x = np.dot(np.dot(beta * o_x, S), xn_sum)

    # compute s_squa_x
    right = np.dot([o_x], S)
    s_squa_x = np.power(beta, -1) + np.dot(right, o_x_T)

    # return the mean and variance of Gaussian distribution
    return [m_x, s_squa_x]





def main():
    filename = stock_code + '_history.csv'
    # ave_r_err = []  # average relative error
    # for i in range(10):
    # print ('File name:', filename)
    filepath = './data/'+ filename
    result = readcsvfile(filepath)
    length = len(result)
    global N
    N = length - 1
    result = np.array(result)
    t = np.arange(N + 1)

    Gaussianpara = bayesiancurvefitting(t[1:], result[:N], N + 1)
    #print ('The true result of time N+1 is: ', result[N + 1])
    print (Gaussianpara[0][0])

    # print absolute mean error and relative error
    # ab_m_err = abs(float(Gaussianpara[0][0]) - float(result[N + 1]))
    # r_err = float(ab_m_err / float(result[N + 1]))
    # print ('The absolute mean error is: ', ab_m_err)
    # print ('The relative error is: ', r_err)
    # ave_r_err.append(r_err)
    # print the average relative error
    # print ('\nThe average relative error is:', average(ave_r_err))


# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
    stock_code = sys.argv[1]
    main()

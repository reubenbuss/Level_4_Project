import math 

def reccuring_probabilty(p,n):
    stop = n//2+n%2
    total = 0
    for i in range(0,stop):
        total += math.comb(n,i)*p**(n-i)*(1-p)**(i)
    return total

#reccuring_probabilty(0.67,40)    
for i in range(40,50):
    if reccuring_probabilty(0.67,i) > 0.99:
        print(i)
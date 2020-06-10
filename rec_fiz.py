#recursive fizzbuzz

def rec_fizz(cur_n, end = 100):
    if cur_n == 100:
        return 
    elif cur_n % 5 == 0 and cur_n % 3 == 0:
        rec_fizz(cur_n+1)
        print('fizzbuzz')
        
    elif cur_n % 5 == 0:
        rec_fizz(cur_n+1)
        print('buzz')
        
    elif cur_n % 3 == 0:
        rec_fizz(cur_n+1)

        print('fizz')
    else:
        rec_fizz(cur_n+1)
        print(cur_n)
  
rec_fizz(1)


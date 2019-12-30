from collector import Collector

c = Collector("cointron-db.cokm27orrtnw.eu-west-2.rds.amazonaws.com","5432","postgres","cointron","tripa!23",
              'PNGEa0YJLxVmPZssX9hDKwu3lhRQmjsyH4bpDTBg7zM2NYYCDoGAR7vtZfQorq8k',
              'kseCG5XF731dbVAwZJHmT3g0po6NjedqyBvUohCnUcZlXhQjxk4B6q4A0jHRfW4C')
c.update_exchange_data()
c.update_training_data()
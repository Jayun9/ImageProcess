import jaydebeapi


class DbProcess:
    def __init__(self, driver, address, id_password, jar_address):
        self.conn = jaydebeapi.connect(driver, address, id_password, jar_address)
        self.cursor = self.conn.cursor()

    def execute(self, query):
        self.cursor.execute(query)
        return self.cursor.fetchall()

    def close(self):
        self.cursor.close()
        self.conn.close()

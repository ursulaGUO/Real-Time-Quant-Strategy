class OrderBook:
    def __init__(self):
        self.list_asks = []
        self.list_bids = []
        self.orders = {}

    def handle_order(self, o):
        if o['action'] == 'new':
            self.handle_new(o)
        elif o['action'] == 'modify':
            self.handle_modify(o)
        elif o['action'] == 'delete':
            self.handle_delete(o)
        else:
            print('Error - Cannot handle this action')

    def handle_new(self, o):
        # Create new order
        new_order = {
            'id': o['id'],
            'price': o['price'],
            'quantity': o['quantity'],
            'side': o['side']
        }

        # Append order to correct list
        if new_order['side'] == 'bid':
            self.list_bids.append(new_order)
            # Sort bids in descending order of price
            self.list_bids.sort(key=lambda x: x['price'], reverse=True)
        else:
            self.list_asks.append(new_order)
            # Sort asks in ascending order of price
            self.list_asks.sort(key=lambda x: x['price'])

        self.orders[new_order['id']] = new_order

    def handle_modify(self, o):
        curr_id = o['id']
        curr_quantity = o['quantity']

        index = self.find_order_in_a_list(o, self.list_asks)
        list_to_lookup = self.list_asks

        if index is None:
            list_to_lookup = self.list_bids
            index = self.find_order_in_a_list(o, self.list_bids)

        if index is not None:
            list_to_lookup[index]['quantity'] = curr_quantity 

        self.orders[curr_id]['quantity'] = curr_quantity  

    def handle_delete(self, o):
        del_order_id = o['id']

        if del_order_id in self.orders:
            del self.orders[del_order_id]

        index = self.find_order_in_a_list(o, self.list_asks)
        list_to_lookup = self.list_asks
        if index is None:
            list_to_lookup = self.list_bids
            index = self.find_order_in_a_list(o, self.list_bids)

        if index is not None:
            del list_to_lookup[index]

    def find_order_in_a_list(self, o, lookup_list):
        lookup_id = o['id']
        for i in range(len(lookup_list)):
            if lookup_list[i]['id'] == lookup_id:
                return i
        return None
# !/usr/bin/env python
# -*- coding: utf-8 -*-


class Node(object):
    def __init__(self, prev, next, value):
        self.prev = prev
        self.next = next
        self.value = value


class DoubleLinkTable(object):

    """ 双向循环链表 """

    def __init__(self, content=None):
        self.count = 0
        if content is not None:
            self.head = Node(None, None, content[0])
            self.head.prev = self.head
            self.head.next = self.head
            self.node = self.head
            self.__create(content[1:])
        else:
            self.head = Node(None, None, None)
            self.head.prev = self.head
            self.head.next = self.head
            self.node = self.head

    def get_node(self, index):
        if index == 0:
            return self.head
        if index < 0 or index > self.count:
            raise Exception("Index out of bound.")

        if index < self.count/2:
            self.node = self.head.next
            i = 0
            while i < index-1:
                self.node = self.node.next
                i += 1
            return self.node

        self.node = self.head.prev
        r_idx = self.count - index
        j = 0
        while j < r_idx:
            self.node = self.node.prev
            j += 1
        return self.node

    def get(self, index):
        return self.get_node(index).value

    def insert(self, index, value):
        cur_node = self.get_node(index)
        new_node = Node(cur_node, cur_node.next, value)
        cur_node.next.prev = new_node
        cur_node.next = new_node
        self.count += 1

    def __create(self, content):
        for i, x in enumerate(content):
            self.insert(i, x)


if __name__ == "__main__":
    loosen = [0, 1, 2, 3, 4, 5]

    link = DoubleLinkTable(loosen)
    print(link.get(0), link.get(1), link.get(2), link.get(3), link.get(4), link.get(5))
    print()
---
layout: post
title: Flutter UI - 채팅앱을 보고 따라해보기 - (2)
author: soonyubing
description:  
featuredImage: null
img: null
tags: flutter chatapp
categories: frontend
date: '2023-03-10 11:17:00 +0900'
# image:
#   src: /assets/img/Chirpy-title-image.jpg
#   width: 1000   # in pixels
#   height: 400   # in pixels
#   alt: image alternative text
---
<style>
r { color: Red }
o { color: Orange }
g { color: Green }
bgr {background-color:Red}
bgy {background-color : Yellow}
bg {background-color: #83f5ef; color : Green}
</style>

![Desktop View](/assets/img/chatpage.png){: width="1200" height="700" }

대부분 코드로 생략하겠음.

몇가지만 보자면, 일단 키보드를 치다가 빈 화면을 눌렀을 때 키보드가 내려가게 하기 위해서 body를 `GestureDetector` 로 감싸고 tap 되었을 때 키보드 controller인 focusNode.unfocus 하도록 하였음.

Emoji picker는 다음 사이트에서 가져올 수 있음. 
configure 값은 demo에 있는 값 그대로 가져온거임. 

새로 알게된 위젯이 있는데, `willpopscope` . 얘는 현재 페이지에서 emoji picker가 켜진채로 뒤로가기 버튼 (안드로이드의 경우) 를 눌렀을 때 emoji picker가 내려가는 게 아니라 페이지가 pop 되는 것을 방지하기 위해 추가함.

```dart
import 'package:emoji_picker_flutter/emoji_picker_flutter.dart';
import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart' as foundation;
import '../Model/chat_model.dart';

class IndividualPage extends StatefulWidget {
  const IndividualPage({super.key, required this.chatModel});
  final ChatModel chatModel;
  @override
  State<IndividualPage> createState() => _IndividualPageState();
}

class _IndividualPageState extends State<IndividualPage> {
  late ChatModel _chatModel;
  TextEditingController _textController = TextEditingController();
  FocusNode _focusNode = FocusNode();
  bool isEmojiVisible = false;
  @override
  void initState() {
    // TODO: implement initState
    super.initState();
    this._chatModel = widget.chatModel;
    _focusNode.addListener(() {
      if (_focusNode.hasFocus) {
        isEmojiVisible = false;
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        leading: Row(children: [
          IconButton(
            icon: Icon(Icons.arrow_back),
            onPressed: () {
              Navigator.pop(context);
            },
          ),
        ]),
        centerTitle: true,
        title: Column(children: [
          Text(_chatModel.name!),
          Text("last seen today at 12:05")
        ]),
        actions: [
          IconButton(onPressed: () {}, icon: Icon(Icons.videocam)),
          IconButton(onPressed: () {}, icon: Icon(Icons.call)),
          PopupMenuButton<String>(onSelected: (value) {
            print(value);
          }, itemBuilder: (BuildContext context) {
            return const [
              PopupMenuItem(child: Text("New Group"), value: "New Group"),
              PopupMenuItem(
                  child: Text("New Broadcast"), value: "New Broadcast"),
              PopupMenuItem(child: Text("Whatsapp Web"), value: "Whatsapp Web"),
              PopupMenuItem(
                  child: Text("Starred Message"), value: "Starred Message"),
              PopupMenuItem(child: Text("Settings"), value: "Settings"),
            ];
          })
        ],
      ),
      body: Container(
        height: MediaQuery.of(context).size.height,
        width: MediaQuery.of(context).size.width,
        child: WillPopScope(
          onWillPop: () {
            if (isEmojiVisible) {
              setState(() {
                isEmojiVisible = false;
              });
            } else {
              Navigator.pop(context);
            }
            return Future.value(false);
          },
          child: GestureDetector(
            onTap: () {
              setState(() {
                isEmojiVisible = false;
                _focusNode.unfocus();
              });
            },
            child: Stack(
              children: [
                ListView(),
                Align(
                  alignment: Alignment.bottomCenter,
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.end,
                    children: [
                      Row(
                        children: [
                          SizedBox(
                            width: MediaQuery.of(context).size.width - 55,
                            child: Card(
                              margin: const EdgeInsets.only(
                                  left: 2, right: 2, bottom: 8),
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(25),
                              ),
                              child: TextFormField(
                                focusNode: _focusNode,
                                controller: _textController,
                                maxLines: 5,
                                minLines: 1,
                                textAlignVertical: TextAlignVertical.center,
                                keyboardType: TextInputType.multiline,
                                decoration: InputDecoration(
                                    border: InputBorder.none,
                                    hintText: "Type a messages",
                                    contentPadding: const EdgeInsets.all(5),
                                    prefixIcon: IconButton(
                                      onPressed: () {
                                        _focusNode.unfocus();
                                        _focusNode.canRequestFocus = false;
                                        setState(() {
                                          isEmojiVisible = !isEmojiVisible;
                                        });
                                        print(isEmojiVisible);
                                      },
                                      icon: const Icon(Icons.emoji_emotions),
                                    ),
                                    suffixIcon: Row(
                                      mainAxisSize: MainAxisSize.min,
                                      children: [
                                        IconButton(
                                          onPressed: () {},
                                          icon: const Icon(Icons.attach_file),
                                        ),
                                        IconButton(
                                          onPressed: () {},
                                          icon: const Icon(Icons.camera_alt),
                                        ),
                                      ],
                                    )),
                              ),
                            ),
                          ),
                          Padding(
                            padding: const EdgeInsets.only(bottom: 8),
                            child: CircleAvatar(
                              child: IconButton(
                                icon: const Icon(Icons.mic),
                                onPressed: () {},
                              ),
                            ),
                          )
                        ],
                      ),
                      EmojiSelect()
                    ],
                  ),
                )
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget EmojiSelect() {
    return Offstage(
      offstage: !isEmojiVisible,
      child: SizedBox(
        height: 250,
        child: EmojiPicker(
          textEditingController: _textController,
          config: Config(
            columns: 7,
            // Issue: https://github.com/flutter/flutter/issues/28894
            emojiSizeMax: 32 *
                (foundation.defaultTargetPlatform == TargetPlatform.iOS
                    ? 1.30
                    : 1.0),
            verticalSpacing: 0,
            horizontalSpacing: 0,
            gridPadding: EdgeInsets.zero,
            initCategory: Category.RECENT,
            bgColor: const Color(0xFFF2F2F2),
            indicatorColor: Colors.blue,
            iconColor: Colors.grey,
            iconColorSelected: Colors.blue,
            backspaceColor: Colors.blue,
            skinToneDialogBgColor: Colors.white,
            skinToneIndicatorColor: Colors.grey,
            enableSkinTones: true,
            showRecentsTab: true,
            recentsLimit: 28,
            replaceEmojiOnLimitExceed: false,
            noRecents: const Text(
              'No Recents',
              style: TextStyle(fontSize: 20, color: Colors.black26),
              textAlign: TextAlign.center,
            ),
            loadingIndicator: const SizedBox.shrink(),
            tabIndicatorAnimDuration: kTabScrollDuration,
            categoryIcons: const CategoryIcons(),
            buttonMode: ButtonMode.MATERIAL,
            checkPlatformCompatibility: true,
          ),
        ),
      ),
    );
  }
}


```